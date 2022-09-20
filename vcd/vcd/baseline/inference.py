#!/usr/bin/env python3
import argparse
import collections
import enum
import tqdm
import logging
from abc import ABC

import torch.cuda
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import glob
import os.path
import itertools
import numpy as np
import subprocess
import tempfile
from typing import List, Tuple, Dict, Any, Iterable
from torchvision.datasets.folder import default_loader
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from vcd.storage import load_features, store_features
from vcd.index import VideoFeature


class InferenceTransforms(enum.Enum):
    SSCD = Compose(
        [
            Resize(288),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ImageDataset:
    """Abstract image dataset."""

    def __init__(self, transform=None, img_transform=None, loader=default_loader):
        self.loader = loader
        self.transform = transform
        self.img_transform = img_transform

    def get_item(self, idx: int) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        filename, record = self.get_item(idx)
        img = self.loader(filename)
        record["input"] = img
        if self.img_transform:
            record["input"] = self.img_transform(record["input"])
        if self.transform:
            record = self.transform(record)
        return record


_ = """
class DistributedIterableDataset(IterableDataset, ABC):

    rank = None
    world_size = None

    def set_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def get_distributed_worker_info(self):
        assert self.rank is not None, "Use DistributedIterableDataset subclasses with DistributedDataLoader"
        worker_info = get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        worker_id += num_workers * self.rank
        num_workers *= self.world_size
        return worker_id, num_workers


class DistributedDataLoader(DataLoader):

    def _get_iterator(self):
        if isinstance(self.dataset, DistributedIterableDataset):
            self.dataset.set_distributed(get_rank(), get_world_size())
        return super()._get_iterator()
"""


class VideoDataset(IterableDataset):
    """Decodes video frames at a fixed FPS via ffmpeg."""

    def __init__(
        self,
        path: str,
        fps: float,
        batch_size=None,
        img_transform=None,
        extensions=["mp4"],
        distributed_rank=0,
        distributed_world_size=1,
    ):
        assert distributed_rank < distributed_world_size
        self.path = path
        self.fps = fps
        self.batch_size = batch_size
        self.img_transform = img_transform
        if len(extensions) == 1:
            filenames = glob.glob(os.path.join(path, f"*.{extensions[0]}"))
        else:
            filenames = glob.glob(os.path.join(path, f"*.*"))
            filenames = (fn for fn in filenames if fn.rsplit(".", 1)[-1] in extensions)
        self.videos = sorted(filenames)
        assert distributed_rank < distributed_world_size
        self.rank = distributed_rank
        self.world_size = distributed_world_size
        self.selected_videos = [
            (i, video)
            for (i, video) in enumerate(self.videos)
            if (i % self.world_size) == self.rank
        ]

    def num_videos(self) -> int:
        return len(self.selected_videos)

    def __iter__(self):
        for i, video in self.selected_videos:
            # logging.info("Reading video %d of %d: %s", i, len(self.videos), os.path.basename(video))
            if self.batch_size:
                frames = self.read_frames(i, video)
                while True:
                    batch = list(itertools.islice(frames, self.batch_size))
                    if not batch:
                        break
                    yield default_collate(batch)
            else:
                yield from self.read_frames(i, video)

    def read_frames(self, video_id, video):
        null = open(os.devnull, "w")
        with tempfile.TemporaryDirectory() as dir:
            video_name = os.path.basename(video)
            subprocess.check_call(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    video,
                    "-start_number",
                    "0",
                    "-q",
                    "0",
                    "-vf",
                    "fps=%f" % self.fps,
                    os.path.join(dir, "%07d.jpg"),
                ],
                stderr=null,
            )
            # logging.info("Reading %s frames", video_name)
            i = 0
            while True:
                frame_fn = os.path.join(dir, f"{i:07d}.jpg")
                # logging.info("Try file %s", frame_fn)
                if not os.path.exists(frame_fn):
                    break
                yield self.read_frame(video_id, video_name, i, frame_fn)
                i += 1
            # logging.info("Got %d frames for %s", i, video_name)

    def read_frame(self, video_id, video_name, frame_id, frame_fn):
        img = default_loader(frame_fn)
        name = os.path.basename(video_name).split(".")[0]
        record = {
            "name": name,
            # "video_id": video_id,
            # "frame_id": frame_id,
            "timestamp": frame_id / self.fps,
            # "instance_id": video_id * 10e8 + frame_id,
        }
        if self.img_transform:
            img = self.img_transform(img)
        record["input"] = img
        return record


parser = argparse.ArgumentParser()
inference_parser = parser.add_argument_group("Inference")
inference_parser.add_argument("--torchscript_path", required=True)
inference_parser.add_argument("--batch_size", type=int, default=32)
inference_parser.add_argument("--distributed_rank", type=int, default=0)
inference_parser.add_argument("--distributed_size", type=int, default=1)
inference_parser.add_argument("--transforms", default=1)
inference_parser.add_argument("--output_path", required=True)

dataset_parser = parser.add_argument_group("Dataset")
dataset_parser.add_argument("--dataset_path", required=True)
dataset_parser.add_argument("--fps", default=1, type=float)
dataset_parser.add_argument("--video_extensions", default="mp4")


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference.py")
logger.setLevel(logging.INFO)


@torch.no_grad()
def run_inference(dataloader, model, device) -> Iterable[VideoFeature]:
    name = None
    embeddings = []
    timestamps = []

    for batch in dataloader:
        names = batch["name"]
        assert names[0] == names[-1]  # single-video batches
        if name is not None and name != names[0]:
            yield VideoFeature(
                video_id=name,
                timestamps=np.concatenate(timestamps),
                feature=np.concatenate(embeddings, axis=0),
            )
            # TODO: do something with it
            embeddings = []
            timestamps = []
        name = names[0]
        img = batch["input"].to(device)
        embeddings.append(model(img).cpu().numpy())
        timestamps.append(batch["timestamp"].numpy())

    yield VideoFeature(
        video_id=name,
        timestamps=np.concatenate(timestamps),
        feature=np.concatenate(embeddings, axis=0),
    )


def main(args):
    logger.info("Loading model")
    model = torch.jit.load(args.torchscript_path)
    model.eval()
    logger.info("Setting up dataset")
    # TODO: configure
    transforms = InferenceTransforms.SSCD.value
    extensions = args.video_extensions.split(",")
    # batch_size = args.batch_size if args.preserve_aspect_ratio else None
    dataset = VideoDataset(
        args.dataset_path,
        fps=args.fps,
        img_transform=transforms,
        batch_size=args.batch_size,
        extensions=extensions,
        distributed_world_size=args.distributed_size,
        distributed_rank=args.distributed_rank,
    )
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model = model.to(device)
    loader = DataLoader(dataset, pin_memory=True, batch_size=None)

    if args.distributed_size > 1:
        worker_output_path = os.path.join(args.output_path, str(args.distributed_rank))
    else:
        worker_output_path = args.output_path
    os.makedirs(worker_output_path, exist_ok=True)

    progress = tqdm.tqdm(total=dataset.num_videos())
    for vf in run_inference(loader, model, device):
        with open(os.path.join(worker_output_path, f"{vf.video_id}.npz"), "wb") as f:
            store_features(f, [vf])
        progress.update()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
