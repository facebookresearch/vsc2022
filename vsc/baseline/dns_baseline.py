#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module implements a baseline matching method based on SSCD features.

Usage:

First, adapt the SSCD "sscd_disc_mixup" torchscript model to remove L2
normalization. See `adapt_sscd_model.py`.

Second, run inference on both the queries and reference datasets, by
calling the inference script on each dataset with the adapted SSCD model.

Finally, run this script to perform retrieval and matching.
"""
import argparse
import logging
import os
import numpy as np
from typing import List, Dict, Tuple

import torch

import matplotlib.pyplot as plt
from vsc.baseline.localization import VCSLLocalizationMaxSim
from vsc.candidates import CandidateGeneration, MaxScoreAggregation
from vsc.index import VideoFeature
from vsc.metrics import (
    average_precision,
    AveragePrecision,
    CandidatePair,
    Dataset,
    evaluate_matching_track,
    Match,
)
from vsc.storage import load_features, store_features, convert_to_dict
from vsc.baseline.dns_index import Accelerator

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dns_baseline.py")
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--torchscript_path",
    help="Path to the fine-grained student model used for similarity calculation.",
    type=str,
    required=True,
)
parser.add_argument(
    "--query_coarse_features",
    help="Path to query coarse descriptors",
    type=str,
    required=True,
)
parser.add_argument(
    "--ref_coarse_features",
    help="Path to reference coarse descriptors",
    type=str,
    required=True,
)
parser.add_argument(
    "--query_fine_features",
    help="Path to query fine descriptors",
    type=str,
    required=True,
)
parser.add_argument(
    "--ref_fine_features",
    help="Path to reference fine descriptors",
    type=str,
    required=True,
)
parser.add_argument(
    "--output_path",
    help="The path to write match prediction.",
    type=str,
    required=True,
)
parser.add_argument(
    "--accelerator",
    help="Device used for the similarity calculation",
    choices=[x.name.lower() for x in Accelerator],
    default="cpu",
    type=str,
)
parser.add_argument(
    "--ground_truth",
    help="Path to the ground truth (labels) CSV file.",
    type=str,
)
parser.add_argument(
    "--overwrite",
    help="Overwrite prediction files, if found.",
    action="store_true",
)


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
        sim = sim / 2.0 + 0.5
        sim = sim.cpu().numpy()

        if self.geometric_mean:
            query = self.queries[candidate.query_id].feature
            ref = self.refs[candidate.ref_id].feature

            sim_cg = np.matmul(query, ref.T) + self.similarity_bias
            sim = np.sqrt(sim.clip(1e-7) * sim_cg.clip(1e-7))
        return sim


def search(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    retrieve_per_query: float = 1200.0,
    candidates_per_query: float = 25.0,
) -> List[CandidatePair]:
    aggregation = MaxScoreAggregation()
    logger.info("Searching")
    cg = CandidateGeneration(refs, aggregation)
    num_to_retrieve = int(retrieve_per_query * len(queries))
    candidates = cg.query(queries, global_k=num_to_retrieve)
    num_candidates = int(candidates_per_query * len(queries))
    candidates = candidates[:num_candidates]
    logger.info("Got %d candidates", len(candidates))
    return candidates


def localize_and_verify(
    model: torch.nn.Module,
    queries_fine: Dict[str, VideoFeature],
    refs_fine: Dict[str, VideoFeature],
    queries_coarse: List[VideoFeature],
    refs_coarse: List[VideoFeature],
    candidates: List[CandidatePair],
    localize_per_query: float = 5.0,
    device: str = "cpu",
) -> List[Match]:
    num_to_localize = int(len(queries_fine) * localize_per_query)
    candidates = candidates[:num_to_localize]

    alignment = VCSLLocalizationDnS(
        model,
        queries_fine,
        refs_fine,
        queries_coarse,
        refs_coarse,
        model_type="TN",
        tn_max_step=5,
        min_length=4,
        concurrency=16,
        similarity_bias=0.5,
        device=device,
    )

    matches = []
    logger.info("Aligning %s candidate pairs", len(candidates))
    BATCH_SIZE = 512
    i = 0
    while i < len(candidates):
        batch = candidates[i : i + BATCH_SIZE]
        matches.extend(alignment.localize_all(batch))
        i += len(batch)
        logger.info(
            "Aligned %d pairs of %d; %d predictions so far",
            i,
            len(candidates),
            len(matches),
        )

    return matches


def match(
    model: torch.nn.Module,
    queries_fine: Dict[str, VideoFeature],
    refs_fine: Dict[str, VideoFeature],
    queries_coarse: List[VideoFeature],
    refs_coarse: List[VideoFeature],
    output_path: str,
    device: str,
) -> Tuple[str, str]:
    # Search
    candidates = search(queries_coarse, refs_coarse)
    os.makedirs(output_path, exist_ok=True)
    candidate_file = os.path.join(output_path, "candidates.csv")
    CandidatePair.write_csv(candidates, candidate_file)

    # Localize and verify
    matches = localize_and_verify(
        model,
        queries_fine,
        refs_fine,
        queries_coarse,
        refs_coarse,
        candidates,
        device=device,
    )
    matches_file = os.path.join(output_path, "matches.csv")
    Match.write_csv(matches, matches_file)
    return candidate_file, matches_file


def create_pr_plot(ap: AveragePrecision, filename: str):
    ap.pr_curve.plot(linewidth=1)
    plt.savefig(filename)
    plt.show()


def main(args):
    if os.path.exists(args.output_path) and not args.overwrite:
        raise Exception(
            f"Output path already exists: {args.output_path}. Do you want to --overwrite?"
        )

    model = torch.jit.load(args.torchscript_path)
    if "fg" != model.student_type:
        raise Exception(
            f"Only fine-grained student are accepted for similarity calculation."
        )

    device = Accelerator[args.accelerator.upper()].get_device()
    model = model.eval().to(device)

    queries_fine = load_features(args.query_fine_features, Dataset.QUERIES)
    queries_fine = convert_to_dict(queries_fine)

    refs_fine = load_features(args.ref_fine_features, Dataset.REFS)
    refs_fine = convert_to_dict(refs_fine)

    queries_coarse = load_features(args.query_coarse_features, Dataset.QUERIES)
    refs_coarse = load_features(args.ref_coarse_features, Dataset.REFS)

    candidate_file, match_file = match(
        model,
        queries_fine,
        refs_fine,
        queries_coarse,
        refs_coarse,
        args.output_path,
        args.accelerator,
    )

    if not args.ground_truth:
        return

    # Descriptor track uAP (approximate)
    gt_matches = Match.read_csv(args.ground_truth, is_gt=True)
    gt_pairs = CandidatePair.from_matches(gt_matches)
    candidate_pairs = CandidatePair.read_csv(candidate_file)
    candidate_uap = average_precision(gt_pairs, candidate_pairs)
    logger.info(f"Candidate uAP: {candidate_uap.ap:.4f}")
    candidate_pr_file = os.path.join(args.output_path, "candidate_precision_recall.pdf")
    create_pr_plot(candidate_uap, candidate_pr_file)

    # Matching track metric:
    match_metrics = evaluate_matching_track(args.ground_truth, match_file)
    logger.info(f"Matching track metric: {match_metrics.segment_ap.ap:.4f}")
    matching_pr_file = os.path.join(args.output_path, "precision_recall.pdf")
    create_pr_plot(match_metrics.segment_ap, matching_pr_file)
    logger.info(f"Candidates: {candidate_file}")
    logger.info(f"Matches: {match_file}")
    logger.info(f"Candidate PR plot: {candidate_pr_file}")
    logger.info(f"Match PR plot: {matching_pr_file}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
