# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch

from gscodec import ptq


def test_pack_roundtrip():
    rng = np.random.default_rng(0)
    for bits in (1, 3, 5, 10, 13):
        v = rng.integers(0, 1 << bits, size=1001)
        assert np.array_equal(ptq.unpack_bits(ptq.pack_bits(v, bits), v.size, bits), v)


def test_segment_bits():
    bits = ptq.segment_bits(102, 32, 5, 10)
    assert bits == 17850
    assert abs(bits / 3 / 1000 - 5.95) < 1e-9


def test_kmeans_codebook():
    torch.manual_seed(0)
    x = torch.randn(5000)
    cb = ptq.Codebook.fit(x.abs() + 1, x, n_bits=4)
    assert cb.n_bits == 4 and cb.scales.numel() == 16
    assert torch.all(cb.weights[1:] >= cb.weights[:-1])
    idx = ptq.quantize_codebook(x, cb.weights)
    err = (ptq.dequantize_codebook(idx, cb.weights) - x).abs().mean()
    assert err < 0.1


def test_uniform_positions():
    x = torch.linspace(0, 224, 57)
    idx = ptq.quantize_uniform(x, 0.0, 224.0, 10)
    assert (ptq.dequantize_uniform(idx, 0.0, 224.0, 10) - x).abs().max() <= 224 / 1023 / 2 + 1e-6
