# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Tuple, Optional

from vsc.candidates import CandidateGeneration, MaxScoreAggregation
from vsc.metrics import (
    average_precision,
    AveragePrecision,
    CandidatePair,
    Dataset,
    Match,
)
from vsc.storage import load_features

logger = logging.getLogger("descriptor_eval_lib.py")
logger.setLevel(logging.INFO)


RETRIEVAL_CANDIDATES_PER_QUERY = 20 * 60  # similar to K=20 for ~60 second videos
AGGREGATED_CANDIDATES_PER_QUERY = 25


def evaluate_descriptor_track(
    query_feature_filename: str,
    ref_feature_filename: str,
    ground_truth_filename: Optional[str],
) -> Tuple[AveragePrecision, List[CandidatePair]]:
    logger.info("Starting Descriptor level eval")
    query_features = load_features(query_feature_filename, Dataset.QUERIES)
    logger.info(f"Loaded {len(query_features)} query features")
    ref_features = load_features(ref_feature_filename, Dataset.REFS)
    logger.info(f"Loaded {len(ref_features)} ref features")

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

    if ground_truth_filename is None:
        return None, score_candidates

    gt_matches = Match.read_csv(ground_truth_filename, is_gt=True)
    gt_pairs = CandidatePair.from_matches(gt_matches)
    logger.info(f"Loaded ground truth from {ground_truth_filename}")
    ap = average_precision(gt_pairs, score_candidates)
    logger.info(f"Descriptor track micro-AP (uAP): {ap.ap:.4f}")

    return ap, score_candidates
