#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Adapt an SSCD model by removing L2 normalization.

This is focused on the sscd_disc_mixup model available here:
https://github.com/facebookresearch/sscd-copy-detection

Some SSCD models have a different structure, and may require different adaptation.
"""
import argparse
import collections
import os

import torch
from torch import nn
from torch.nn import functional as F


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_torchscript",
    help="Path to the SSCD torchscript model to adapt.",
    type=str,
    required=True,
)
parser.add_argument(
    "--output_torchscript",
    help="The adapted SSCD model to write.",
    type=str,
    required=True,
)


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)


def check_model_equivalence(model1, model2, distance=1e-3):
    input_tensor = torch.randn([2, 3, 64, 64])
    with torch.no_grad():
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)
    distances = (output1 - output2).pow(2).sum(dim=1)
    if (distances > distance).sum().item():
        raise Exception("Distances not all within expected tolerance")


def remove_l2_norm(sscd_script_model):
    sscd_script_model = sscd_script_model.eval()
    embeddings = sscd_script_model.embeddings
    if embeddings.original_name == "L2Norm":
        # SSCD Classy Vision model
        modules = [("backbone", sscd_script_model.backbone)]
    else:
        # SSCD Torchvision model
        components = list(embeddings.children())
        names = [c.original_name for c in components]
        assert names == ["GlobalGeMPool2d", "Linear", "L2Norm"]
        modules = [
            ("backbone", sscd_script_model.backbone),
            ("pool", components[0]),
            ("project", components[1]),
        ]
    model = nn.Sequential(collections.OrderedDict(modules))
    # The new model followed by L2 norm is equivalent to the original model
    check_model_equivalence(nn.Sequential(model, L2Norm()), sscd_script_model)
    input_tensor = torch.randn([2, 3, 64, 64])
    script_model = torch.jit.trace(model, input_tensor)
    # The torchscript model is equivalent to the new model
    check_model_equivalence(model, script_model)
    return script_model


def main(args):
    if os.path.exists(args.output_torchscript):
        raise Exception("Output file already exists")
    sscd_model = torch.jit.load(args.input_torchscript)
    script_model = remove_l2_norm(sscd_model)
    torch.jit.save(script_model, args.output_torchscript)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
