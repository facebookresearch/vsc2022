#!/usr/bin/env python3
"""
Descriptor track evaluation script.
"""
import logging
from argparse import ArgumentParser, Namespace

from vsc.baseline.candidates import CandidateGeneration, MaxScoreAggregation
from vsc.metrics import average_precision, CandidatePair, Match
from vsc.storage import load_features

parser = ArgumentParser()
parser.add_argument(
    "--query_features",
    help="Path containing query features",
    type=str,
    required=True,
)
parser.add_argument(
    "--ref_features",
    help="Path containing reference features",
    type=str,
    required=True,
)
parser.add_argument(
    "--candidates_output",
    help="Path to write candidates (optional)",
    type=str,
)
parser.add_argument(
    "--ground_truth", help="Path containing Groundtruth", type=str, required=True
)


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("descriptor_eval.py")
logger.setLevel(logging.INFO)


RETRIEVAL_CANDIDATES_PER_QUERY = 20 * 60  # similar to K=20 for ~60 second videos
AGGREGATED_CANDIDATES_PER_QUERY = 25


def main(args: Namespace):
    logger.info("Starting Descriptor level eval")
    query_features = load_features(args.query_features, expected_prefix="Q")
    logger.info(f"Loaded {len(query_features)} query features")
    ref_features = load_features(args.ref_features, expected_prefix="R")
    logger.info(f"Loaded {len(ref_features)} ref features")
    gt_matches = Match.read_csv(args.ground_truth, is_gt=True)
    gt_pairs = CandidatePair.from_matches(gt_matches)
    logger.info(f"Loaded ground truth from {args.ground_truth}")

    # TODO: require a fixed number of input videos per track.
    # TODO: emit threshold that the search uses
    retrieval_candidates = int(RETRIEVAL_CANDIDATES_PER_QUERY * len(query_features))
    num_candidates = int(AGGREGATED_CANDIDATES_PER_QUERY * len(query_features))

    logger.info(f"Performing search for {retrieval_candidates} nearest vectors")
    cg = CandidateGeneration(ref_features, MaxScoreAggregation())
    candidates = cg.query(query_features, global_k=retrieval_candidates)
    logger.info(f"Got {len(candidates)} unique video pairs.")
    if len(candidates) > num_candidates:
        logger.info(f"Limiting to {num_candidates} highest score pairs.")
        score_candidates = candidates[:num_candidates]
    else:
        score_candidates = candidates

    ap = average_precision(gt_pairs, score_candidates)
    logger.info(f"Descriptor track micro-AP (uAP): {ap.ap:.4f}")

    if args.candidates_output:
        logger.info(f"Storing candidates to {args.candidates_output}")
        CandidatePair.write_csv(candidates, args.candidates_output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
