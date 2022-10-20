from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple

from PIL import Image

ImageT = Image.Image


class VideoReader(ABC):
    def __init__(self, video_path: str, required_fps: float) -> None:
        self.video_path = video_path
        self.required_fps = required_fps
        self.original_fps = max(1, self.fps) if self.fps else 1
        self.video_frames = None

    @property
    @abstractmethod
    def fps(self) -> Optional[float]:
        pass

    @abstractmethod
    def frames(self) -> Iterable[Tuple[float, ImageT]]:
        """
        returns a tuple of [timestamp, Image]
        """
        pass
