# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Optional

import numpy as np
from vsc.index import VideoFeature
from vsc.metrics import Dataset, format_video_id


def store_features(f, features: List[VideoFeature], dataset: Optional[Dataset] = None):
    video_ids = []
    feats = []
    timestamps = []
    for feature in features:
        video_id = format_video_id(feature.video_id, dataset)
        video_ids.append(np.full(len(feature), video_id))
        feats.append(feature.feature)
        timestamps.append(feature.timestamps)
    video_ids = np.concatenate(video_ids)
    feats = np.concatenate(feats)
    timestamps = np.concatenate(timestamps)
    np.savez(f, video_ids=video_ids, features=feats, timestamps=timestamps)


def same_value_ranges(values):
    start = 0
    value = values[start]

    for i, v in enumerate(values):
        if v == value:
            continue
        yield value, start, i
        start = i
        value = values[start]

    yield value, start, len(values)


def load_features(f, dataset: Optional[Dataset] = None):
    data = np.load(f, allow_pickle=False)
    video_ids = data["video_ids"]
    feats = data["features"]
    timestamps = data["timestamps"]

    ts_dims = len(timestamps.shape)
    if timestamps.shape[0] != feats.shape[0]:
        raise ValueError(
            f"Expected the same number of timestamps as features: got "
            f"{timestamps.shape[0]} timestamps for {feats.shape[0]} features"
        )
    if not (ts_dims == 1 or timestamps.shape[1:] == (2,)):
        print(f"timestamps.shape[1:]: {timestamps.shape[1:]}")
        print(f"timestamps.shape[1:] == [2]: {timestamps.shape[1:] == [2]}")
        raise ValueError(f"Unexpected timestamp shape. Got {timestamps.shape}")

    results = []
    for video_id, start, end in same_value_ranges(video_ids):
        video_id = format_video_id(video_id, dataset)
        item = VideoFeature(
            video_id=video_id,
            timestamps=timestamps[start:end],
            feature=feats[start:end, :],
        )
        results.append(item)
    return results


def convert_to_dict(features: List[VideoFeature]) -> Dict[str, VideoFeature]:
    return {m.video_id: m for m in features}
