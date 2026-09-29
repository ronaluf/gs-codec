# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Post-training quantization: k-means codebooks for scales and amplitudes, uniform grid for centers."""

import typing as tp

import numpy as np
import torch


def fit_kmeans_1d(x: torch.Tensor, k: int, n_iter: int = 40) -> torch.Tensor:
    """1D k-means with quantile initialization; returns ``k`` sorted centroids."""
    x = x.flatten().float().cpu()
    centers = torch.quantile(x, torch.linspace(0.5 / k, 1 - 0.5 / k, k))
    for _ in range(n_iter):
        assign = (x.unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)
        new_centers = centers.clone()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = x[mask].mean()
        if torch.allclose(new_centers, centers, atol=1e-5):
            break
        centers = new_centers
    return centers.sort().values


class Codebook(tp.NamedTuple):
    """Scalar codebooks for one bit depth."""
    scales: torch.Tensor
    weights: torch.Tensor

    @property
    def n_bits(self) -> int:
        return int(self.scales.numel()).bit_length() - 1

    def to_dict(self) -> tp.Dict[str, tp.List[float]]:
        return {"scales": self.scales.tolist(), "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.List[float]]) -> "Codebook":
        return cls(torch.tensor(d["scales"], dtype=torch.float32),
                   torch.tensor(d["weights"], dtype=torch.float32))

    @classmethod
    def fit(cls, scales: torch.Tensor, weights: torch.Tensor, n_bits: int) -> "Codebook":
        k = 1 << n_bits
        return cls(fit_kmeans_1d(scales, k), fit_kmeans_1d(weights, k))


def codebook_key(mode: str, freeze_positions: bool, n_gaussians: int, n_bits: int) -> str:
    """Codebook identifier, e.g. ``predictor/free/ng102/b5``."""
    return f"{mode}/{'grid' if freeze_positions else 'free'}/ng{n_gaussians}/b{n_bits}"


def quantize_codebook(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Nearest-centroid indices."""
    c = centers.to(x.device).view(-1)
    return (x.unsqueeze(-1) - c).abs().argmin(dim=-1)


def dequantize_codebook(idx: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return centers.to(idx.device)[idx]


def quantize_uniform(x: torch.Tensor, vmin: float, vmax: float, n_bits: int) -> torch.Tensor:
    levels = (1 << n_bits) - 1
    return torch.round((x.clamp(vmin, vmax) - vmin) / (vmax - vmin) * levels).long()


def dequantize_uniform(idx: torch.Tensor, vmin: float, vmax: float, n_bits: int) -> torch.Tensor:
    levels = (1 << n_bits) - 1
    return idx.float() / levels * (vmax - vmin) + vmin


def segment_bits(n_gaussians: int, dimension: int, n_bits: int, position_bits: int = 0) -> int:
    """Payload bits of one segment."""
    return n_gaussians * n_bits + dimension * n_gaussians * n_bits + n_gaussians * position_bits


def pack_bits(values: np.ndarray, n_bits: int) -> bytes:
    """Pack non-negative integers, MSB first, ``n_bits`` each."""
    v = np.asarray(values, dtype=np.uint32).ravel()
    if v.size and int(v.max()) >= (1 << n_bits):
        raise ValueError(f"Value does not fit in {n_bits} bits.")
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.uint32)
    bits = ((v[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(bits.ravel()).tobytes()


def unpack_bits(data: bytes, count: int, n_bits: int) -> np.ndarray:
    """Inverse of :func:`pack_bits`."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), count=count * n_bits)
    bits = bits.reshape(count, n_bits).astype(np.uint32)
    weights = (1 << np.arange(n_bits - 1, -1, -1)).astype(np.uint32)
    return (bits * weights).sum(axis=1).astype(np.int64)
