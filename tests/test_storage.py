# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from vsc.index import VideoFeature
from vsc.metrics import Dataset
from vsc.storage import load_features, store_features


class StorageTest(unittest.TestCase):
    dims = 32

    def fake_timestamps(self, length: float, fps: float):
        return np.arange(length) / fps

    def fake_vf(self, video_id, length, fps=1.0):
        embeddings = np.random.randn(length, self.dims)
        timestamps = self.fake_timestamps(length, fps)
        return VideoFeature(
            video_id=video_id, timestamps=timestamps, feature=embeddings
        )

    def test_storage(self):
        features = [
            self.fake_vf(2, 10),
            self.fake_vf(3, 20, fps=3.0),
            self.fake_vf(1, 30, fps=0.5),
        ]
        with tempfile.NamedTemporaryFile() as f:
            store_features(f, features, Dataset.QUERIES)
            f.flush()
            restored = load_features(f.name)

        features.sort(key=lambda x: x.video_id)

        self.assertEqual(len(features), len(restored))
        for a, b in zip(features, restored):
            self.assertEqual(f"Q{a.video_id:06d}", b.video_id)
            assert_allclose(b.timestamps, a.timestamps)
            assert_allclose(b.feature, a.feature)

        # Test storing features with string IDs
        with tempfile.NamedTemporaryFile() as f:
            store_features(f, restored)  # no dataset needed
            f.flush()
            restored2 = load_features(f.name)

        for a, b in zip(restored, restored2):
            self.assertEqual(a.video_id, b.video_id)
            assert_allclose(b.timestamps, a.timestamps)
            assert_allclose(b.feature, a.feature)

    def test_out_of_order(self):
        features = [
            self.fake_vf("Q000002", 10),
            self.fake_vf("Q000003", 20, fps=3.0),
            self.fake_vf("Q000001", 30, fps=0.5),
        ]
        with tempfile.NamedTemporaryFile() as f:
            store_features(f, features, Dataset.QUERIES)
            f.flush()
            data = np.load(f.name, allow_pickle=False)

        # Scramble the order.
        N = data["features"].shape[0]
        order = np.random.permutation(N)
        data = {
            "features": data["features"][order, :],
            "video_ids": data["video_ids"][order],
            "timestamps": data["timestamps"][order],
        }
        with tempfile.NamedTemporaryFile() as f:
            np.savez(f, **data)
            f.flush()
            restored = load_features(f.name)

        features.sort(key=lambda x: x.video_id)
        restored.sort(key=lambda x: x.video_id)
        self.assertEqual(len(features), len(restored))

        for a, b in zip(features, restored):
            self.assertEqual(a.video_id, b.video_id)
            assert_allclose(b.timestamps, a.timestamps)
            assert_allclose(b.feature, a.feature)


class IntervalStorageTest(StorageTest):
    def fake_timestamps(self, length: float, fps: float):
        timestamps = super().fake_timestamps(length, fps)
        return np.stack([timestamps, timestamps + fps], axis=1)
