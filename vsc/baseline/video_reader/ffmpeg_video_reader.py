# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import tempfile
from typing import Iterable, Optional, Tuple

from PIL import Image
from torchvision.datasets.folder import default_loader

from vsc.baseline.video_reader.video_reader import VideoReader

ImageT = Image.Image


class FFMpegVideoReader(VideoReader):
    def __init__(self, video_path: str, required_fps: float, ffmpeg_path: str):
        self.ffmpeg_path = ffmpeg_path
        super().__init__(video_path, required_fps)

    @property
    def fps(self) -> Optional[float]:
        return None

    def frames(self) -> Iterable[Tuple[float, float, ImageT]]:
        with tempfile.TemporaryDirectory() as dir, open(os.devnull, "w") as null:
            subprocess.check_call(
                [
                    self.ffmpeg_path,
                    "-nostdin",
                    "-y",
                    "-i",
                    self.video_path,
                    "-start_number",
                    "0",
                    "-q",
                    "0",
                    "-vf",
                    "fps=%f" % self.required_fps,
                    os.path.join(dir, "%07d.png"),
                ],
                stderr=null,
            )
            i = 0
            while True:
                frame_fn = os.path.join(dir, f"{i:07d}.png")
                if not os.path.exists(frame_fn):
                    break
                img = default_loader(frame_fn)
                i += 1
                yield ((i - 1) / self.original_fps, i / self.original_fps, img)
