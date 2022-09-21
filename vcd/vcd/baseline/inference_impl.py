#!/usr/bin/env python3
import glob
import itertools
import logging
import os
import subprocess
import tempfile
from typing import Iterable

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from vcd.baseline.inference import Accelerator, InferenceTransforms
from vcd.index import VideoFeature
from vcd.storage import store_features


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference_impl.py")
logger.setLevel(logging.INFO)


def build_transforms(transform: InferenceTransforms):
    return {
        InferenceTransforms.SSCD: transforms.Compose(
            [
                transforms.Resize(288),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    }[transform]


class VideoDataset(IterableDataset):
    """Decodes video frames at a fixed FPS via ffmpeg."""

    def __init__(
        self,
        path: str,
        fps: float,
        batch_size=None,
        img_transform=None,
        extensions=("mp4",),
        distributed_rank=0,
        distributed_world_size=1,
        ffmpeg_path="ffmpeg",
    ):
        assert distributed_rank < distributed_world_size
        self.path = path
        self.fps = fps
        self.batch_size = batch_size
        self.img_transform = img_transform
        self.ffmpeg_path = ffmpeg_path
        if len(extensions) == 1:
            filenames = glob.glob(os.path.join(path, f"*.{extensions[0]}"))
        else:
            filenames = glob.glob(os.path.join(path, "*.*"))
            filenames = (fn for fn in filenames if fn.rsplit(".", 1)[-1] in extensions)
        self.videos = sorted(filenames)
        if not self.videos:
            raise Exception("No videos found!")
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
        with tempfile.TemporaryDirectory() as dir, open(os.devnull, "w") as null:
            video_name = os.path.basename(video)
            subprocess.check_call(
                [
                    self.ffmpeg_path,
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


def should_use_cuda(args) -> bool:
    accelerator = Accelerator[args.accelerator.upper()]
    return accelerator == Accelerator.CUDA


def get_device(args, rank, world_size):
    if should_use_cuda(args):
        assert torch.cuda.is_available()
        num_devices = torch.cuda.device_count()
        if args.processes > num_devices:
            raise Exception(
                f"Asked for {args.processes} processes and cuda, but only "
                f"{num_devices} devices found"
            )
        if args.processes > 1 or world_size <= num_devices:
            device_num = rank
        else:
            device_num = 0
        torch.cuda.set_device(device_num)
        return torch.device("cuda", device_num)
    return torch.device("cpu")


def worker_process(args, rank, world_size):
    logger.info(f"Starting worker {rank} of {world_size}.")
    device = get_device(args, rank, world_size)
    logger.info("Loading model")
    model = torch.jit.load(args.torchscript_path)
    model.eval()
    logger.info("Setting up dataset")
    transforms = build_transforms(InferenceTransforms[args.transforms])
    extensions = args.video_extensions.split(",")
    dataset = VideoDataset(
        args.dataset_path,
        fps=args.fps,
        img_transform=transforms,
        batch_size=args.batch_size,
        extensions=extensions,
        distributed_world_size=world_size,
        distributed_rank=rank,
        ffmpeg_path=args.ffmpeg_path,
    )
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=None, pin_memory=device.type == "cuda")
    worker_output_path = os.path.join(args.output_path, str(rank))
    os.makedirs(worker_output_path, exist_ok=True)

    progress = tqdm.tqdm(total=dataset.num_videos())
    for vf in run_inference(loader, model, device):
        with open(os.path.join(worker_output_path, f"{vf.video_id}.npz"), "wb") as f:
            store_features(f, [vf])
        progress.update()


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
