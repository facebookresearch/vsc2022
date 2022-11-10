#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Descriptor track evaluation script.
"""
import logging
from argparse import ArgumentParser, Namespace

from vsc.descriptor_eval_lib import evaluate_descriptor_track
from vsc.metrics import CandidatePair

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
parser.add_argument("--ground_truth", help="Path containing Groundtruth", type=str)


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("descriptor_eval_lib.py")
logger.setLevel(logging.INFO)


def main(args: Namespace):
    ap, candidates = evaluate_descriptor_track(
        args.query_features, args.ref_features, args.ground_truth
    )

    if args.candidates_output:
        logger.info(f"Storing candidates to {args.candidates_output}")
        CandidatePair.write_csv(candidates, args.candidates_output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
