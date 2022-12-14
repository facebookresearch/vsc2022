# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

from vsc.candidates import CandidateGeneration, MaxScoreAggregation
from vsc.index import VideoFeature
from vsc.metrics import CandidatePair


class CandidateGenerationTest(unittest.TestCase):
    def test_candidate_generation(self):
        queries = [
            VideoFeature(
                video_id=1,
                feature=np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                ),
                timestamps=np.array([0.0, 1.0, 2.0]),
            ),
        ]
        refs = [
            VideoFeature(
                video_id=5,
                feature=np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 2, 0],
                        [0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
                timestamps=np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
            ),
            VideoFeature(
                video_id=8,
                feature=np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                    dtype=np.float32,
                ),
                timestamps=np.array([[0.0, 5.0], [5.0, 10.0], [10.0, 15.0]]),
            ),
            VideoFeature(
                video_id=10,
                feature=np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0.25],
                        [0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
                timestamps=np.array([0.0, 0.1, 0.2]),
            ),
        ]

        cg = CandidateGeneration(refs, MaxScoreAggregation())
        candidates = cg.query(queries, 2 * 3)

        self.assertEqual(3, len(candidates))
        self.assertEqual(
            candidates,
            [
                CandidatePair(query_id=1, ref_id=5, score=2.0),
                CandidatePair(query_id=1, ref_id=8, score=1.0),
                CandidatePair(query_id=1, ref_id=10, score=0.25),
            ],
        )


if __name__ == "__main__":
    unittest.main()
