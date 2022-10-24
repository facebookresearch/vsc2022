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
from typing import Callable, List, Tuple

import faiss  # @manual
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from vsc.baseline.candidates import CandidateGeneration, MaxScoreAggregation
from vsc.baseline.localization import (
    VCSLLocalizationCandidateScore,
    VCSLLocalizationMaxSim,
)
from vsc.index import VideoFeature
from vsc.metrics import (
    average_precision,
    AveragePrecision,
    CandidatePair,
    evaluate_matching_track,
    Match,
)
from vsc.storage import load_features, store_features


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
    help="Path to query descriptors",
    type=str,
    required=True,
)
parser.add_argument(
    "--ref_features",
    help="Path to reference descriptors",
    type=str,
    required=True,
)
parser.add_argument(
    "--score_norm_features",
    help="Path to score normalization descriptors",
    type=str,
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


def transform_features(
    features: List[VideoFeature], transform: Callable
) -> List[VideoFeature]:
    return [
        dataclasses.replace(feature, feature=transform(feature.feature))
        for feature in features
    ]


def localize_and_verify(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    candidates: List[CandidatePair],
    localize_per_query: float = 5.0,
    score_normalization: bool = False,
) -> List[Match]:
    num_to_localize = int(len(queries) * localize_per_query)
    candidates = candidates[:num_to_localize]

    if score_normalization:
        alignment = VCSLLocalizationMaxSim(
            queries,
            refs,
            model_type="TN",
            tn_max_step=5,
            min_length=4,
            concurrency=16,
            similarity_bias=0.5,
        )
    else:
        alignment = VCSLLocalizationCandidateScore(
            transform_features(queries, normalize),
            transform_features(refs, normalize),
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
    score_normalization: bool = False,
) -> Tuple[str, str]:
    # Search
    candidates = search(queries, refs)
    os.makedirs(output_path, exist_ok=True)
    candidate_file = os.path.join(output_path, "candidates.csv")
    CandidatePair.write_csv(candidates, candidate_file)

    # Localize and verify
    matches = localize_and_verify(
        queries,
        refs,
        candidates,
        score_normalization=score_normalization,
    )
    matches_file = os.path.join(output_path, "matches.csv")
    Match.write_csv(matches, matches_file)
    return candidate_file, matches_file


def score_normalize(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    score_norm_refs: List[VideoFeature],
    l2_normalize: bool = True,
    replace_dim: bool = True,
    beta: float = 1.2,
) -> Tuple[List[VideoFeature], List[VideoFeature]]:
    """
    CSLS style score normalization (as used in the Image Similarity Challenge)
    has the following form. We compute a bias term for each query:

      bias(query) = - beta * sim(query, noise)

    then compute score normalized similarity by incorporating this as an
    additive term for each query:

      sim_sn(query, ref) = sim(query, ref) + bias(query)

    sim(query, ref) is inner product similarity (query * ref), and
    sim(query, noise) is some function of query similarity to a noise dataset
    (score_norm_refs here), such as the similarity to the nearest neighbor.

    We encode the bias term as an extra dimension in the query descriptor,
    and add a constant 1 dimension to reference descriptors, so that inner-
    product similarity is the score-normalized similarity:

      query' = [query bias(query)]
      ref' = [ref 1]
      query' * ref' = (query * ref) + (bias(query) * 1)
          = sim(query, ref) + bias(query) = sim_sn(query, ref)
    """
    if {f.video_id for f in refs}.intersection({f.video_id for f in score_norm_refs}):
        raise Exception(
            "Normalizing on the dataset we're evaluating on is against VSC rules. "
            "An independent dataset is needed."
        )
    if score_norm_refs is not None and replace_dim:
        # Make space for the additional score normalization dimension.
        # We could also use PCA dim reduction, but re-centering can be
        # destructive.
        logger.info("Replacing dimension")
        sn_features = np.concatenate([ref.feature for ref in score_norm_refs], axis=0)
        low_var_dim = sn_features.var(axis=0).argmin()
        queries, refs, score_norm_refs = [
            transform_features(
                x, lambda feature: np.delete(feature, low_var_dim, axis=1)
            )
            for x in [queries, refs, score_norm_refs]
        ]
    if l2_normalize:
        logger.info("L2 normalizing")
        queries, refs, score_norm_refs = [
            transform_features(x, normalize) for x in [queries, refs, score_norm_refs]
        ]
    logger.info("Applying score normalization")
    index = CandidateGeneration(score_norm_refs, MaxScoreAggregation()).index.index
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)

    adapted_queries = []
    # Add the additive normalization term to the queries as an extra dimension.
    for query in queries:
        # KNN search is ok here (versus a threshold/radius/range search) since
        # we're not searching the dataset we're evaluating on.
        similarity, ids = index.search(query.feature, 1)
        norm_term = -beta * similarity[:, :1]
        feature = np.concatenate([query.feature, norm_term], axis=1)
        adapted_queries.append(dataclasses.replace(query, feature=feature))
    adapted_refs = []
    for ref in refs:
        ones = np.ones_like(ref.feature[:, :1])
        feature = np.concatenate([ref.feature, ones], axis=1)
        adapted_refs.append(dataclasses.replace(ref, feature=feature))
    return adapted_queries, adapted_refs


def create_pr_plot(ap: AveragePrecision, filename: str):
    ap.pr_curve.plot(linewidth=1)
    plt.savefig(filename)
    plt.show()


def main(args):
    if os.path.exists(args.output_path) and not args.overwrite:
        raise Exception(
            f"Output path already exists: {args.output_path}. Do you want to --overwrite?"
        )
    queries = load_features(args.query_features)
    refs = load_features(args.ref_features)
    score_normalization = False
    if args.score_norm_features:
        queries, refs = score_normalize(
            queries,
            refs,
            load_features(args.score_norm_features),
        )
        score_normalization = True
        os.makedirs(args.output_path, exist_ok=True)
        store_features(os.path.join(args.output_path, "sn_queries.npz"), queries)
        store_features(os.path.join(args.output_path, "sn_refs.npz"), refs)
    candidate_file, match_file = match(
        queries,
        refs,
        args.output_path,
        score_normalization=score_normalization,
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
