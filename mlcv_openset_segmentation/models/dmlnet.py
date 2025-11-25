"""
Code adapted from the official implementation of:
Cen et al., "Deep Metric Learning for Open World Semantic Segmentation"
Source: https://github.com/cennavi/DMLNet
"""

import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv3x3_bn_relu(in_channels, out_channels, stride=1):
    """3x3 convolution + BatchNorm + ReLU."""
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.inplanes = 128

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResnetDilated(nn.Module):
    def __init__(self, backbone, dilate_scale=8):
        super().__init__()

        if dilate_scale == 8:
            backbone.layer3.apply(partial(self._apply_dilation, dilate=2))
            backbone.layer4.apply(partial(self._apply_dilation, dilate=4))
        elif dilate_scale == 16:
            backbone.layer4.apply(partial(self._apply_dilation, dilate=2))
        else:
            raise ValueError("dilate_scale must be 8 or 16")

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu1 = backbone.relu1
        self.conv2 = backbone.conv2
        self.bn2 = backbone.bn2
        self.relu2 = backbone.relu2
        self.conv3 = backbone.conv3
        self.bn3 = backbone.bn3
        self.relu3 = backbone.relu3
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    @staticmethod
    def _apply_dilation(module, dilate):
        if isinstance(module, nn.Conv2d):
            if module.stride == (2, 2):
                module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.dilation = (dilate // 2,) * 2
                    module.padding = (dilate // 2,) * 2
            else:
                if module.kernel_size == (3, 3):
                    module.dilation = (dilate,) * 2
                    module.padding = (dilate,) * 2

    def forward(self, x, return_feature_maps=False):
        features = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        if return_feature_maps:
            return features
        return [x]


class PPMDeepSupEmbedding(nn.Module):
    def __init__(
        self,
        num_classes=13,
        in_channels=2048,
        use_softmax=False,
        pool_scales=(1, 2, 3, 6),
        center_magnitude: float = 3.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_softmax = use_softmax

        # PPM branches
        self.ppm = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
                for scale in pool_scales
            ]
        )

        self.cbr_deepsup = conv3x3_bn_relu(in_channels // 2, in_channels // 4)
        self.conv_last_deepsup = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)
        self.dropout_deepsup = nn.Dropout2d(0.1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_scales) * 512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

        # Class centers matrix (num_classes x num_classes)
        centers = torch.eye(num_classes) * center_magnitude
        self.register_buffer("centers", centers, persistent=False)

    def forward(self, features, seg_size=None, return_features=True):
        conv5 = features[-1]
        batch_size, _, height, width = conv5.shape

        ppm_out = [conv5]
        for pool in self.ppm:
            ppm_out.append(
                F.interpolate(
                    pool(conv5),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        ppm_out = torch.cat(ppm_out, dim=1)
        fused_features = ppm_out.clone()

        logits = self.conv_last(ppm_out)

        # Metric embedding distance computation
        reshaped = logits.permute(0, 2, 3, 1).contiguous()
        reshaped = reshaped.view(batch_size, -1, self.num_classes)

        expanded = reshaped.unsqueeze(2)
        expanded = expanded.expand(-1, -1, self.num_classes, -1)

        distances = expanded - self.centers
        dist_to_center = -torch.sum(distances**2, dim=3)

        logits = (
            dist_to_center.permute(0, 2, 1)
            .contiguous()
            .view(batch_size, self.num_classes, height, width)
        )

        if self.use_softmax:
            logits = F.interpolate(
                logits, size=seg_size, mode="bilinear", align_corners=False
            )

            if return_features:
                fused_features = F.interpolate(
                    fused_features,
                    size=seg_size,
                    mode="bilinear",
                    align_corners=False,
                )
                return logits, fused_features
            else:
                return logits

        # Deep supervision branch
        conv4 = features[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        if return_features:
            return (logits, _), fused_features
        else:
            return (logits, _)


class ModelBuilder:

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            module.bias.data.fill_(1e-4)

    @staticmethod
    def build_encoder(weights_path="", dilate_scale=8):
        base = resnet50()
        encoder = ResnetDilated(base, dilate_scale=dilate_scale)

        if weights_path:
            state = torch.load(Path(weights_path), map_location="cpu")
            encoder.load_state_dict(state, strict=False)

        return encoder

    @staticmethod
    def build_decoder(
        in_channels=512,
        num_classes=13,
        weights_path="",
        use_softmax=False,
        center_magnitude=3.0,
    ):
        decoder = PPMDeepSupEmbedding(
            num_classes=num_classes,
            in_channels=in_channels,
            use_softmax=use_softmax,
            center_magnitude=center_magnitude,
        )
        decoder.apply(ModelBuilder.initialize_weights)

        if weights_path:
            state = torch.load(Path(weights_path), map_location="cpu")
            decoder.load_state_dict(state, strict=False)

        return decoder
