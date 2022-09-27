from typing import List

import numpy as np
from vcd.index import VideoFeature


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


def load_features(f) -> List[VideoFeature]:
    data = np.load(f, allow_pickle=False)
    video_ids = data["video_ids"]
    feats = data["features"]
    timestamps = data["timestamps"]

    results = []
    for video_id, start, end in same_value_ranges(video_ids):
        results.append(
            VideoFeature(
                video_id=video_id,
                timestamps=timestamps[start:end],
                feature=feats[start:end, :],
            )
        )
    return results
