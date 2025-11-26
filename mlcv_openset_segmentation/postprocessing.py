"""
Code adapted from the official implementation of:
Jung et al., "Standardized Max Logits: A Simple yet Effective Approach for Identifying
Unexpected Road Obstacles in Urban-Scene Segmentation".
Source: https://github.com/shjung13/Standardized-max-logits
"""

import math

import numpy as np
import torch
import torch.nn as nn
from kornia.morphology import dilation, erosion
from scipy import ndimage as ndi


def build_structuring_elements(
    max_radius: int = 9,
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selem = torch.ones((3, 3), device=device, dtype=torch.float32)
    selem_dilation = torch.as_tensor(
        ndi.generate_binary_structure(2, 1),
        dtype=torch.float32,
        device=device,
    )

    dilation_kernels = {}

    for radius in range(1, max_radius + 1):
        size = 2 * radius + 1
        kernel = torch.zeros((1, 1, size, size), device=device, dtype=torch.float32)
        # center pixel
        kernel[:, :, radius, radius] = 1.0

        # apply dilation 'radius' times
        for _ in range(radius):
            kernel = dilation(kernel, selem_dilation)

        dilation_kernels[radius] = kernel.squeeze(0).squeeze(0)

    return selem, selem_dilation, dilation_kernels


SELEM, SELEM_DILATION, DILATION_KERNELS = build_structuring_elements()


def find_boundaries(label: torch.Tensor) -> torch.Tensor:
    """
    Compute boundary mask as difference between dilated and eroded maps.
    """
    assert label.ndim == 4, f"Expected 4D tensor (B, C, H, W), got {label.shape}"
    dilated = dilation(label.float(), SELEM_DILATION)
    eroded = erosion(label.float(), SELEM)
    boundaries = (dilated != eroded).float()
    return boundaries


def expand_boundaries(boundaries: torch.Tensor, radius: int = 0) -> torch.Tensor:
    """
    Expand boundary maps by a given radius.
    """
    if radius == 0:
        return boundaries

    kernel = DILATION_KERNELS.get(radius)
    expanded_boundaries = dilation(boundaries, kernel)
    return expanded_boundaries


class BoundarySuppressionWithSmoothing(nn.Module):
    """
    Apply boundary suppression and dilated smoothing.
    """

    def __init__(
        self,
        boundary_suppression: bool = True,
        boundary_width: int = 4,
        boundary_iteration: int = 4,
        dilated_smoothing: bool = True,
        kernel_size: int = 7,
        dilation: int = 6,
    ):
        super().__init__()

        self.boundary_suppression = boundary_suppression
        self.boundary_width = boundary_width
        self.boundary_iteration = boundary_iteration
        self.dilated_smoothing = dilated_smoothing
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Gaussian kernel for smoothing
        sigma = 1.0
        size = 7
        gaussian_kernel = np.fromfunction(
            lambda x, y: (1 / (2 * math.pi * sigma**2))
            * math.e
            ** (
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)
                / (2 * sigma**2)
            ),
            (size, size),
        )
        gaussian_kernel /= np.sum(gaussian_kernel)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).float()
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)

        # 3x3 box filter (for local averaging)
        self.first_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=False,
        )
        self.first_conv.weight = nn.Parameter(torch.ones_like((self.first_conv.weight)))

        # Dilated Gaussian smoothing
        self.second_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=1,
            dilation=self.dilation,
            bias=False,
        )
        self.second_conv.weight = nn.Parameter(gaussian_kernel)

    def forward(
        self,
        x: torch.Tensor,
        prediction: torch.Tensor = None,
    ) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)

        assert x.ndim == 4, f"Expected 4D tensor (B, 1, H, W), got {x.shape}"

        out = x

        if self.boundary_suppression:
            boundaries = find_boundaries(prediction.unsqueeze(1))

            if self.boundary_iteration != 0:
                assert self.boundary_width % self.boundary_iteration == 0
                step = self.boundary_width // self.boundary_iteration

            expanded_boundaries = None

            for iteration in range(self.boundary_iteration):
                if out.ndim != 4:
                    out = out.unsqueeze(1)

                prev_out = out

                if self.boundary_width == 0 or iteration == self.boundary_iteration - 1:
                    expansion_width = 0
                else:
                    expansion_width = self.boundary_width - step * iteration - 1

                expanded_boundaries = expand_boundaries(
                    boundaries,
                    radius=expansion_width,
                )

                non_boundary_mask = (expanded_boundaries == 0).float()

                # Boundary masking and padding
                filter_radius = 1  # for 3x3
                pad = filter_radius
                x_masked = out * non_boundary_mask

                pad_layer = nn.ReplicationPad2d(pad)
                x_padded = pad_layer(x_masked)
                mask_padded = pad_layer(non_boundary_mask)

                # Sum over receptive field
                y = self.first_conv(x_padded)
                num_valid = self.first_conv(mask_padded).long()

                # Average, preserving original where no valid pixels
                avg_y = torch.where(
                    (num_valid == 0),
                    prev_out,
                    y / num_valid,
                )
                out = avg_y

                # Only update boundary regions
                out = torch.where(non_boundary_mask == 0, out, prev_out)

                del expanded_boundaries, non_boundary_mask

            # Second stage: dilated smoothing
            if self.dilated_smoothing:
                pad = self.dilation * 3
                out = nn.ReplicationPad2d(pad)(out)
                out = self.second_conv(out)

            return out.squeeze(1)

        # No boundary suppression, optional smoothing only
        if self.dilated_smoothing:
            pad = self.dilation * 3
            out = nn.ReplicationPad2d(pad)(out)
            out = self.second_conv(out)
        else:
            out = x

        return out.squeeze(1)
