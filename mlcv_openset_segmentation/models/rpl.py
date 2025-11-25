"""
Code adapted from the official implementation of:
Liu et al., "Residual Pattern Learning for Pixel-wise Out-of-Distribution Detection
in Semantic Segmentation".
Source: https://github.com/yyliu01/RPL
"""

import torch
import torch.nn as nn
from torchvision import transforms
from copy import deepcopy


class EnergyScore(nn.Module):
    def __init__(self, segmenter, use_gaussian=True):
        super().__init__()
        self.segmenter = segmenter.eval()
        self.use_gaussian = use_gaussian

    @torch.no_grad()
    def forward(self, inputs):

        _, score = self.segmenter(inputs)
        anomaly_score = -torch.logsumexp(score, dim=1)

        if self.use_gaussian:
            anomaly_score = transforms.GaussianBlur(7, sigma=1)(
                anomaly_score.unsqueeze(0)
            ).squeeze(0)

        return score, anomaly_score


class EnergyEntropyScore(nn.Module):

    def __init__(self, segmenter, use_gaussian=True):
        super().__init__()
        self.segmenter = segmenter.eval()
        self.use_gaussian = use_gaussian

    @torch.no_grad()
    def forward(self, inputs):

        _, score = self.segmenter(inputs)

        prob = torch.softmax(score, dim=1)
        entropy = -torch.sum(prob * torch.log(prob), dim=1)
        energy = -torch.logsumexp(score, dim=1)

        anomaly_score = energy + entropy

        if self.use_gaussian:
            anomaly_score = transforms.GaussianBlur(7, sigma=1)(
                anomaly_score.unsqueeze(0)
            ).squeeze(0)

        return score, anomaly_score


class RPLDeepLab(nn.Module):
    def __init__(self, segmenter):
        super().__init__()

        self.encoder = self._clone(segmenter.encoder, unfreeze=False)
        self.decoder = self._clone(segmenter.decoder, unfreeze=False)

        self.final = nn.Sequential(
            self._clone(segmenter.decoder.block2, unfreeze=False),
            self._clone(segmenter.segmentation_head, unfreeze=False),
        )

        self.atten_aspp_final = nn.Conv2d(256, 304, kernel_size=1, bias=False)

        self.residual_anomaly_block = nn.Sequential(
            self._clone(segmenter.decoder.aspp, unfreeze=True),
            self._clone(segmenter.decoder.up, unfreeze=True),
            self.atten_aspp_final,
        )

    def _clone(self, layer, unfreeze=True):
        clone = deepcopy(layer)
        for p in clone.parameters():
            p.requires_grad = unfreeze
        return clone

    def forward(self, x):

        features = self.encoder(x)

        # "vanilla pathway"
        aspp = self.decoder.aspp(features[-1])
        up = self.decoder.up(aspp)
        high_res = self.decoder.block1(features[2])
        concat = torch.cat([up, high_res], dim=1)

        # RPL anomaly residual
        res = self.residual_anomaly_block(features[-1])

        out1 = self.final(concat)
        out2 = self.final(concat + res)

        return out1, out2
