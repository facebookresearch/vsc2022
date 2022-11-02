from typing import List, Union

import numpy as np
from vsc.index import VideoFeature


def store_features(f, features: List[VideoFeature]):
    video_ids = []
    feats = []
    timestamps = []
    for feature in features:
        video_ids.append(np.full(len(feature), feature.video_id))
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


def video_id_int(video_id: Union[str, int], expected_prefix=None) -> int:
    try:
        return int(video_id)
    except ValueError:
        pass
    if isinstance(video_id, str):
        if video_id[0].isalpha() and video_id[1:].isdigit():
            prefix = video_id[0]
            if expected_prefix and prefix != expected_prefix:
                raise ValueError(
                    f"Expected video IDs to begin with {expected_prefix}: got {video_id}"
                )
            return int(video_id[1:])
        return int(video_id)
    raise ValueError(f"Unexpected video id: {video_id}")


def load_features(f, expected_prefix=None) -> List[VideoFeature]:
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
        video_id = video_id_int(video_id, expected_prefix=expected_prefix)
        results.append(
            VideoFeature(
                video_id=video_id,
                timestamps=timestamps[start:end],
                feature=feats[start:end, :],
            )
        )
    return results
