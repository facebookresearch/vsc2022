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
import dataclasses
import argparse
import logging
import enum
import os
from typing import List

import torch
from tqdm import tqdm

from vsc.baseline.score_normalization import score_normalize, transform_features
from vsc.index import VideoFeature
from vsc.metrics import Dataset

from vsc.storage import load_features, store_features
from vsc.baseline.dns.students import CoarseGrainedStudent, FineGrainedStudent


class Student(enum.Enum):
    FINE_ATT = enum.auto()
    FINE_BIN = enum.auto()
    COARSE = enum.auto()

    def get_model(
        self,
    ):
        return self._get_config(self)

    def _get_config(self, value):
        return {
            self.FINE_ATT: FineGrainedStudent(attention=True, pretrained=True),
            self.FINE_BIN: FineGrainedStudent(binarization=True, pretrained=True),
            self.COARSE: CoarseGrainedStudent(pretrained=True),
        }[value]


class Accelerator(enum.Enum):
    CPU = enum.auto()
    CUDA = enum.auto()

    def get_device(
        self,
    ):
        return self._get_config(self)

    def _get_config(self, value):
        return {
            self.CPU: torch.device("cpu"),
            self.CUDA: torch.device("cuda"),
        }[value]


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dns_index.py")
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
    "--accelerator",
    choices=[x.name.lower() for x in Accelerator],
    default="cpu",
    type=str,
)
parser.add_argument(
    "--student",
    help="DnS student used for indexing.",
    choices=[x.name.lower() for x in Student],
    type=str,
    required=True,
)


@torch.no_grad()
def index_videos(
    model: torch.nn.Module,
    features: List[VideoFeature],
    device: torch.device,
) -> List[VideoFeature]:
    indexed_features = []
    for video in tqdm(features):
        feature = torch.from_numpy(video.feature).to(device)
        if model.student_type == "cg":
            feature = feature.unsqueeze(1)
        feature = model.index_video(feature.float())
        if model.student_type == "fg":
            feature = feature > 0 if model.fg_type == "bin" else feature.half()
        feature = feature.cpu().numpy()
        indexed_features.append(dataclasses.replace(video, feature=feature))
    return indexed_features


def main(args):
    if "fine" in args.student and args.score_norm_features:
        raise Exception(
            f"Student type {args.student} can not be combined with score normalization."
        )

    student = Student[args.student.upper()]
    device = Accelerator[args.accelerator.upper()].get_device()

    model = student.get_model().eval().to(device)
    extension = model.get_network_name()

    logger.info(f"Loading query features from {args.query_features}")
    queries = load_features(args.query_features, Dataset.QUERIES)
    logger.info(f"{len(queries)} queries loaded")
    logger.info(f"Index query features based on {model.get_network_name()}")
    indexed_queries = index_videos(model, queries, device)

    logger.info(f"Loading ref features from {args.ref_features}")
    refs = load_features(args.ref_features, Dataset.REFS)
    logger.info(f"{len(refs)} refs loaded")
    logger.info(f"Index ref features based on {model.get_network_name()}")
    indexed_refs = index_videos(model, refs, device)

    if args.score_norm_features:
        logger.info(
            f"Loading features for score normalization from {args.score_norm_features}"
        )
        sn_refs = load_features(args.score_norm_features, Dataset.REFS)
        logger.info(f"{len(sn_refs)} features loaded")
        logger.info(
            f"Index score normalization features based on {model.get_network_name()}"
        )
        sn_refs = index_videos(model, sn_refs, device)

        indexed_queries, indexed_refs = score_normalize(
            indexed_queries,
            indexed_refs,
            sn_refs,
            replace_dim=False,
            beta=1.2,
        )
        extension += "_sn"
    os.makedirs(args.output_path, exist_ok=True)
    store_features(
        os.path.join(args.output_path, f"queries_{extension}.npz"), indexed_queries
    )
    store_features(
        os.path.join(args.output_path, f"refs_{extension}.npz"), indexed_refs
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
