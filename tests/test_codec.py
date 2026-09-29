# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy

import pytest
import torch

from gscodec import GSCode, GSCodec, build_model, ptq

CONFIG = {
    "sample_rate": 24000,
    "channels": 1,
    "segment_duration": 3.0,
    "seanet": {"dimension": 32, "channels": 1, "n_filters": 8, "n_residual_layers": 1, "ratios": [8, 5, 4, 2],
               "activation": "snake", "norm": "weight_norm", "lstm": 2, "pad_mode": "constant",
               "encoder": {}, "decoder": {}},
    "gaussian_splat": {"n_gaussians": 40, "shared_positions": True, "shared_scales": True, "freeze_bias": True,
                       "normalize_input": False, "amortized_predictor": True, "amortized_d_model": 32,
                       "amortized_n_heads": 4, "amortized_n_blocks": 2, "amortized_n_decoder_layers": 1,
                       "amortized_variable_ng": True, "amortized_mode_sampling": True,
                       "amortized_pos_zero_init": False},
    "iterative": {"n_iters": 5, "lr": 0.05, "lr_position_mult": 2.0, "lr_schedule": "cosine",
                  "lr_final_ratio": 0.01, "lr_warmup_iters": 1},
    "defaults": {"mode": "predictor", "n_gaussians": 24, "n_bits": 4, "freeze_positions": False,
                 "position_bits": 10},
}


@pytest.fixture(scope="module")
def codec():
    torch.manual_seed(0)
    c = GSCodec(build_model(copy.deepcopy(CONFIG)), copy.deepcopy(CONFIG))
    z = c.model.encoder(torch.randn(4, 1, c.segment_length) * 0.1)
    for mode in ("predictor", "iterative"):
        for frozen in (False, True):
            p = c.fit(z, mode, 24, frozen)
            c.codebooks[ptq.codebook_key(mode, frozen, 24, 4)] = ptq.Codebook.fit(p["scales"], p["weights"], 4)
    return c


@pytest.mark.parametrize("mode", ["predictor", "iterative"])
@pytest.mark.parametrize("frozen", [False, True])
def test_roundtrip(codec, mode, frozen):
    wav = torch.randn(1, 24000 * 4) * 0.1
    code = codec.encode(wav, mode=mode, freeze_positions=frozen)
    assert code.n_segments == 2 and code.n_gaussians == 24
    assert (code.positions is None) == frozen
    restored = GSCode.from_bytes(code.to_bytes())
    assert torch.equal(restored.scales, code.scales) and torch.equal(restored.weights, code.weights)
    assert restored.mode == mode and restored.length == wav.shape[-1]
    out = codec.decode(restored)
    assert out.shape == wav.shape and torch.isfinite(out).all()
    expected_kbps = ptq.segment_bits(24, 32, 4, 0 if frozen else 10) / 3 / 1000
    assert abs(code.kbps - expected_kbps) < 1e-9
    assert len(code.to_bytes()) * 8 >= code.num_bits


def test_resampling(codec):
    wav = torch.randn(2, 16000) * 0.1
    out = codec.reconstruct(wav, sample_rate=16000)
    assert out.shape == (1, 24000)


def test_save_load(codec, tmp_path):
    codec.save_pretrained(tmp_path)
    loaded = GSCodec.from_pretrained(tmp_path)
    wav = torch.randn(1, 24000) * 0.1
    a = codec.encode(wav, mode="predictor")
    b = loaded.encode(wav, mode="predictor")
    assert torch.equal(a.weights, b.weights) and torch.equal(a.positions, b.positions)


def test_missing_codebook(codec):
    with pytest.raises(KeyError):
        codec.encode(torch.zeros(1, 24000), n_bits=7)
