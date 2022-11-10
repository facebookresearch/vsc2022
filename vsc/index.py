# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
from dataclasses import dataclass
from typing import Iterable, List, NamedTuple, Tuple

import faiss  # @manual
import numpy as np
from faiss.contrib import exhaustive_search  # @manual

SearchIndices = Tuple[int, int, float]


@dataclass
class VideoMetadata:
    video_id: str
    timestamps: np.ndarray  # either Nx2 (start and end timestamps) or N

    def __len__(self):
        return self.timestamps.shape[0]

    def get_timestamps(self, idx: int) -> Tuple[float, float]:
        t = self.timestamps[idx]
        if len(self.timestamps.shape) == 1:
            return (t, t)
        return (t[0], t[1])


@dataclass
class VideoFeature(VideoMetadata):
    feature: np.ndarray

    def __post_init__(self):
        assert self.feature.shape[0] == len(
            self.timestamps
        ), "Mismatched timestamps / feature size"

    def metadata(self):
        return VideoMetadata(video_id=self.video_id, timestamps=self.timestamps)

    def dimensions(self):
        return self.feature.shape[1]


class PairMatch(NamedTuple):
    query_timestamps: Tuple[float, float]
    ref_timestamps: Tuple[float, float]
    score: float


@dataclass
class PairMatches:
    query_id: str
    ref_id: str
    matches: List[PairMatch]

    def records(self):
        for match in self.matches:
            yield {
                "query_id": self.query_id,
                "ref_id": self.ref_id,
                "query_start": match.query_timestamps[0],
                "query_end": match.query_timestamps[1],
                "ref_start": match.ref_timestamps[0],
                "ref_end": match.ref_timestamps[1],
                "score": match.score,
            }


class VideoIndex:
    def __init__(
        self,
        dim: int,
        codec_str: str = "Flat",
        metric: int = faiss.METRIC_INNER_PRODUCT,
    ):
        self.dim = dim
        self.index = faiss.index_factory(self.dim, codec_str, metric)
        self.video_clip_idx = []
        self.video_clip_to_video_ids = []
        self.video_metadata = {}

    def add(self, db: List[VideoFeature]):
        for vf in db:
            self.video_clip_idx.extend(list(range(vf.feature.shape[0])))
            self.video_clip_to_video_ids.extend(
                [vf.video_id for _ in range(vf.feature.shape[0])]
            )
            self.video_metadata[vf.video_id] = vf.metadata()
            self.index.add(vf.feature)

    def search(
        self,
        queries: List[VideoFeature],
        global_k: int,
    ) -> List[PairMatches]:
        query_ids = []
        query_indices = []
        for q in queries:
            query_ids.extend([q.video_id] * len(q))
            query_indices.extend(range(len(q)))
        query_metadatas = {q.video_id: q.metadata() for q in queries}
        query_features = np.concatenate([q.feature for q in queries])
        if global_k < 0:
            # Negative values cause us to use vanilla KNN search
            k = -global_k
            logging.warn(
                "Using local k for KNN search. Warning: this is against the "
                "VSC rules, since predictions for a query-ref pair are not "
                "independent of other references. KNN search is provided for "
                "comparison."
            )
            search_indices = self._knn_search(query_features, k)
        else:
            search_indices = self._global_threshold_knn_search(query_features, global_k)

        pair_nns = collections.defaultdict(list)

        for i, j, score in search_indices:
            query_id = query_ids[i]
            query_idx = query_indices[i]
            query_metadata = query_metadatas[query_id]
            ref_id = self.video_clip_to_video_ids[j]
            ref_idx = self.video_clip_idx[j]
            ref_metadata = self.video_metadata[ref_id]
            match = PairMatch(
                query_timestamps=query_metadata.get_timestamps(query_idx),
                ref_timestamps=ref_metadata.get_timestamps(ref_idx),
                score=score,
            )
            pair_nns[query_id, ref_id].append(match)

        return [
            PairMatches(query_id, ref_id, matches)
            for ((query_id, ref_id), matches) in pair_nns.items()
        ]

    def _global_threshold_knn_search(
        self, query_features: np.ndarray, global_k: int
    ) -> Iterable[SearchIndices]:
        use_similarity = self.index.metric_type == faiss.METRIC_INNER_PRODUCT
        initial_radius = -1e10 if use_similarity else 1e10
        _, limits, similarity, indices = exhaustive_search.range_search_max_results(
            self.index,
            exhaustive_search.exponential_query_iterator(query_features),
            initial_radius,
            max_results=2 * global_k,
            min_results=global_k,
            ngpu=-1,  # use GPU if available
        )
        nq = query_features.shape[0]
        search_indices = []

        for i in range(nq):
            for j in range(limits[i], limits[i + 1]):
                search_indices.append((i, indices[j], similarity[j]))

        search_indices.sort(key=lambda x: x[2], reverse=use_similarity)
        if len(search_indices) > global_k:
            search_indices = search_indices[:global_k]
        return search_indices

    def _knn_search(self, query_features: np.ndarray, k) -> Iterable[SearchIndices]:
        index = self.index
        if faiss.get_num_gpus() > 0:
            logging.info("Moving index to GPU")
            index = faiss.index_cpu_to_all_gpus(self.index)

        logging.info("Performing KNN search")
        similarity, ids = index.search(query_features, k)
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                yield (i, ids[i, j], similarity[i, j])
