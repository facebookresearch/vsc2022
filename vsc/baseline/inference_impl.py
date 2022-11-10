# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import itertools
import logging
import os
from typing import Iterable, List

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from vsc.baseline.inference import Accelerator, InferenceTransforms, VideoReaderType
from vsc.baseline.video_reader.ffmpeg_video_reader import FFMpegVideoReader
from vsc.index import VideoFeature
from vsc.storage import load_features, store_features


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference_impl.py")
logger.setLevel(logging.INFO)


def build_transforms(transform: InferenceTransforms):
    return {
        InferenceTransforms.RESIZE_288: transforms.Compose(
            [
                transforms.Resize(288),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        InferenceTransforms.RESIZE_320_CENTER: transforms.Compose(
            [
                transforms.Resize(320),
                transforms.CenterCrop(320),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
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
        video_reader=VideoReaderType.FFMPEG,
        ffmpeg_path="ffmpeg",
    ):
        assert distributed_rank < distributed_world_size
        self.path = path
        self.fps = fps
        self.batch_size = batch_size
        self.img_transform = img_transform
        self.video_reader = video_reader
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
        video_name = os.path.basename(video)
        name = os.path.basename(video_name).split(".")[0]
        if self.video_reader == VideoReaderType.FFMPEG:
            reader = FFMpegVideoReader(
                video_path=video, required_fps=self.fps, ffmpeg_path=self.ffmpeg_path
            )
        else:
            raise ValueError(f"VideoReaderType: {self.video_reader} not supported")
        for start_timestamp, end_timestamp, frame in reader.frames():
            if self.img_transform:
                frame = self.img_transform(frame)
            record = {
                "name": name,
                "timestamp": np.array([start_timestamp, end_timestamp]),
                "input": frame,
            }
            yield record


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


def worker_process(args, rank, world_size, output_filename):
    logger.info(f"Starting worker {rank} of {world_size}.")
    device = get_device(args, rank, world_size)
    logger.info("Loading model")
    model = torch.jit.load(args.torchscript_path)
    model.eval()
    logger.info("Setting up dataset")
    transforms = build_transforms(InferenceTransforms[args.transforms])
    extensions = args.video_extensions.split(",")
    video_reader = VideoReaderType[args.video_reader.upper()]
    dataset = VideoDataset(
        args.dataset_path,
        fps=args.fps,
        img_transform=transforms,
        batch_size=args.batch_size,
        extensions=extensions,
        distributed_world_size=world_size,
        distributed_rank=rank,
        video_reader=video_reader,
        ffmpeg_path=args.ffmpeg_path,
    )
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=None, pin_memory=device.type == "cuda")

    progress = tqdm.tqdm(total=dataset.num_videos())
    vfs = []
    for vf in run_inference(loader, model, device):
        vfs.append(vf)
        progress.update()

    del loader
    del model
    del dataset

    logger.info(f"Storing worker {rank} outputs")
    store_features(output_filename, vfs)
    logger.info(
        f"Wrote worker {rank} features for {len(vfs)} videos to {output_filename}"
    )


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
                timestamps=np.concatenate(timestamps, axis=0),
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
        timestamps=np.concatenate(timestamps, axis=0),
        feature=np.concatenate(embeddings, axis=0),
    )


def merge_feature_files(filenames: List[str], output_filename: str) -> int:
    features = []
    for fn in filenames:
        features.extend(load_features(fn))
    store_features(output_filename, features)
    return len(features)
