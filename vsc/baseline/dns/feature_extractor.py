import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from einops import rearrange


class DnSResNet50(nn.Module):
    def __init__(self, whitening=True, dims=512):
        super().__init__()

        self.backbone = models.resnet50(pretrained=True)

        self.layers = {"layer1": 28, "layer2": 14, "layer3": 6, "layer4": 3}
        if whitening or dims != 3840:
            from .layers import PCALayer

            self.pca = PCALayer(n_components=dims)

    def extract_region_vectors(self, x):
        tensors = []
        for nm, module in self.backbone._modules.items():
            if nm not in {"avgpool", "fc", "classifier"}:
                x = module(x).contiguous()
                if nm in self.layers:
                    s = self.layers[nm]
                    region_vectors = F.max_pool2d(x, [s, s], int(np.ceil(s / 2)))
                    region_vectors = F.normalize(region_vectors, p=2, dim=1)
                    tensors.append(region_vectors)
        for i in range(len(tensors)):
            tensors[i] = F.normalize(
                F.adaptive_max_pool2d(tensors[i], tensors[-1].shape[2:]), p=2, dim=1
            )
        x = torch.cat(tensors, 1)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        return F.normalize(x, p=2, dim=-1)

    def forward(self, x):
        x = self.extract_region_vectors(x)
        if hasattr(self, "pca"):
            x = self.pca(x)
        return x


class DINO(nn.Module):
    def __init__(self, backbone="dino_vitb8", pooling_param=4.0):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dino:main",
            backbone,
        )
        self.pooling_param = pooling_param

    def _dino_pooling(self, x):
        cls_token = x[:, 0]
        x = x[:, 1:]

        x = F.lp_pool1d(x.clamp(min=1e-6), self.pooling_param, x.size(2))
        x = rearrange(x, "b c r -> b (c r)")

        cls_token = F.normalize(cls_token, p=2, dim=-1)
        x = F.normalize(x, p=2, dim=-1)

        x = torch.cat([cls_token, x], dim=-1)
        return F.normalize(x, p=2, dim=-1)

    def forward(self, x):
        x = self.backbone.get_intermediate_layers(x)[-1]
        x = self._dino_pooling(x)
        return x
