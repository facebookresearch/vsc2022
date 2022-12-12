# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import torch
from typing import List

import numpy as np

from vsc.index import VideoFeature
from vsc.metrics import CandidatePair, Match


class Localization(abc.ABC):
    @abc.abstractmethod
    def localize(self, candidate: CandidatePair) -> List[Match]:
        pass

    def localize_all(self, candidates: List[CandidatePair]) -> List[Match]:
        matches = []
        for candidate in candidates:
            matches.extend(self.localize(candidate))
        return matches


class LocalizationWithMetadata(Localization):
    def __init__(self, queries: List[VideoFeature], refs: List[VideoFeature]):
        self.queries = {m.video_id: m for m in queries}
        self.refs = {m.video_id: m for m in refs}

    def similarity(self, candidate: CandidatePair):
        a = self.queries[candidate.query_id].feature
        b = self.refs[candidate.ref_id].feature
        return np.matmul(a, b.T)


class VCSLLocalization(LocalizationWithMetadata):
    def __init__(self, queries, refs, model_type, similarity_bias=0.0, **kwargs):
        super().__init__(queries, refs)

        # Late import: allow OSS use without VCSL installed
        from vcsl.vta import build_vta_model  # @manual

        self.model = build_vta_model(model_type, **kwargs)
        self.similarity_bias = similarity_bias

    def similarity(self, candidate: CandidatePair):
        """Add an optional similarity bias.

        Some localization methods do not tolerate negative values well.
        """
        return super().similarity(candidate) + self.similarity_bias

    def localize_all(self, candidates: List[CandidatePair]) -> List[Match]:
        sims = [(f"{c.query_id}-{c.ref_id}", self.similarity(c)) for c in candidates]
        results = self.model.forward_sim(sims)
        assert len(results) == len(candidates)
        matches = []
        for (candidate, (key, sim), result) in zip(candidates, sims, results):
            query: VideoFeature = self.queries[candidate.query_id]
            ref: VideoFeature = self.refs[candidate.ref_id]
            assert key == result[0]
            for box in result[1]:
                (x1, y1, x2, y2) = box
                match = Match(
                    query_id=candidate.query_id,
                    ref_id=candidate.ref_id,
                    query_start=query.get_timestamps(x1)[0],
                    query_end=query.get_timestamps(x2)[1],
                    ref_start=ref.get_timestamps(y1)[0],
                    ref_end=ref.get_timestamps(y2)[1],
                    score=0.0,
                )
                score = self.score(candidate, match, box, sim)
                match = match._replace(score=score)
                matches.append(match)
        return matches

    def localize(self, candidate: CandidatePair) -> List[Match]:
        return self.localize_all([candidate])

    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        return 1.0


class VCSLLocalizationMaxSim(VCSLLocalization):
    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        x1, y1, x2, y2 = box
        return similarity[x1:x2, y1:y2].max() - self.similarity_bias


class VCSLLocalizationCandidateScore(VCSLLocalization):
    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        return candidate.score


class VCSLLocalizationDnS(VCSLLocalizationMaxSim):
    def __init__(
        self,
        model,
        queries_fine,
        refs_fine,
        queries_coarse,
        refs_coarse,
        model_type,
        device,
        symmetric=True,
        geometric_mean=True,
        **kwargs,
    ):
        super().__init__(queries_coarse, refs_coarse, model_type, **kwargs)

        self.queries_fine = queries_fine
        self.refs_fine = refs_fine

        self.sim_model = model
        self.device = device

        self.symmetric = symmetric
        self.geometric_mean = geometric_mean

    def _rescale_binaries(self, x):
        if "bin" in self.sim_model.fg_type:
            x = 2 * x - 1
        return x

    @torch.no_grad()
    def similarity(self, candidate: CandidatePair):

        query = self.queries_fine[candidate.query_id].feature
        ref = self.refs_fine[candidate.ref_id].feature

        query = torch.from_numpy(query).to(self.device).float()
        ref = torch.from_numpy(ref).to(self.device).float()

        query = self._rescale_binaries(query)
        ref = self._rescale_binaries(ref)

        sim = self.sim_model(query, ref)
        if self.symmetric:
            simT = self.sim_model(ref, query).mT
            sim = (sim + simT) / 2.0
        sim = sim.cpu().numpy()

        if self.geometric_mean:
            query = self.queries[candidate.query_id].feature
            ref = self.refs[candidate.ref_id].feature

            sim_cg = np.matmul(query, ref.T) + self.similarity_bias
            sim = np.sqrt(sim.clip(1e-7) * sim_cg.clip(1e-7))
        return sim
