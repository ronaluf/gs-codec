# Copyright (c) 2023-present, Descript.
# Modifications copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE.dac file in the root directory of this source tree.
"""Snake activation, x + sin^2(alpha * x) / alpha (Ziyin et al., 2020)."""

import torch
import torch.nn as nn


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """Snake activation with learnable per-channel frequency parameter alpha.

    Args:
        channels (int): Number of channels. Each channel gets its own alpha.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)
