#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Matching track evaluation script.
"""
import logging
from argparse import ArgumentParser, Namespace

from vsc.metrics import evaluate_matching_track


parser = ArgumentParser()
parser.add_argument(
    "--predictions",
    help="Path containing match predictions",
    type=str,
    required=True,
)
parser.add_argument(
    "--ground_truth",
    help="Path containing ground truth labels",
    type=str,
    required=True,
)


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("matching_eval.py")
logger.setLevel(logging.INFO)


def main(args: Namespace):
    metrics = evaluate_matching_track(args.ground_truth, args.predictions)
    segment_ap = metrics.segment_ap.ap
    print(f"Matching track segment AP: {segment_ap:.4f}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
