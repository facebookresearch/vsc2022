import unittest

import faiss  # @manual

import numpy as np

from vsc.index import VideoFeature, VideoIndex


class IndexTest(unittest.TestCase):
    def run_video_index_test(self, global_k: int):
        # test_feature = np.random.rand(50, 50, 32)
        test_feature = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
                [[111, 112, 113], [114, 115, 116], [117, 118, 119]],
            ],
            dtype=np.float32,
        )
        query = [
            VideoFeature(
                video_id=f"Q{idx:06d}",
                feature=feature,
                timestamps=np.arange(3, dtype=np.float32),
            )
            for idx, feature in enumerate(test_feature)
        ]
        db = [
            VideoFeature(
                video_id=f"R{idx:06d}",
                feature=feature,
                timestamps=np.arange(3, dtype=np.float32),
            )
            for idx, feature in enumerate(test_feature)
        ]

        index = VideoIndex(3, "Flat", faiss.METRIC_L2)
        index.add(db)
        results = index.search(query, global_k)
        for result in results:
            self.assertEqual(result.query_id[1:], result.ref_id[1:])

    def test_global_candidate_search(self):
        self.run_video_index_test(1)

    def test_knn_search(self):
        self.run_video_index_test(-1)
