#!/usr/bin/env python3
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
import dataclasses
import logging
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from vsc.baseline.candidates import CandidateGeneration, MaxScoreAggregation
from vsc.baseline.localization import VCSLLocalizationCandidateScore
from vsc.index import VideoFeature
from vsc.metrics import (
    average_precision,
    AveragePrecision,
    CandidatePair,
    evaluate_matching_track,
    Match,
)
from vsc.storage import load_features


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sscd_baseline.py")
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--query_features",
    help="Path to the SSCD torchscript model to adapt.",
    type=str,
    required=True,
)
parser.add_argument(
    "--ref_features",
    help="The adapted SSCD model to write.",
    type=str,
    required=True,
)
parser.add_argument(
    "--output_path",
    help="The path to write match predictions.",
    type=str,
    required=True,
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


def l2_normalize_features(features: List[VideoFeature]) -> List[VideoFeature]:
    return [
        dataclasses.replace(feature, feature=normalize(feature.feature))
        for feature in features
    ]


def localize_and_verify(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    candidates: List[CandidatePair],
    localize_per_query: float = 25.0,
    l2_normalize: bool = True,
) -> List[Match]:
    if l2_normalize:
        queries = l2_normalize_features(queries)
        refs = l2_normalize_features(refs)

    num_to_localize = int(len(queries) * localize_per_query)
    candidates = candidates[:num_to_localize]

    alignment = VCSLLocalizationCandidateScore(
        queries,
        refs,
        model_type="TN",
        tn_max_step=5,
        min_length=4,
        concurrency=16,
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
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    output_path: str,
) -> Tuple[str, str]:
    # Search
    candidates = search(queries, refs)
    os.makedirs(output_path, exist_ok=True)
    candidate_file = os.path.join(output_path, "candidates.csv")
    CandidatePair.write_csv(candidates, candidate_file)

    # Localize and verify
    matches = localize_and_verify(queries, refs, candidates)
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
    # TODO: complexity
    queries = load_features(args.query_features)
    refs = load_features(args.ref_features)
    candidate_file, match_file = match(queries, refs, args.output_path)

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
    logger.info(f"Matching track metric: {match_metrics.segment_ap_v2.ap:.4f}")
    matching_pr_file = os.path.join(args.output_path, "precision_recall.pdf")
    create_pr_plot(match_metrics.segment_ap_v2, matching_pr_file)
    logger.info(f"Candidates: {candidate_file}")
    logger.info(f"Matches: {match_file}")
    logger.info(f"Candidate PR plot: {candidate_pr_file}")
    logger.info(f"Match PR plot: {matching_pr_file}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
