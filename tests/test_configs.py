# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from gscodec.models import builders

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")


@pytest.mark.parametrize("solver", ["compression/gaussian_24khz", "compression/gs_predictor_24khz"])
def test_training_configs_build(solver):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=[f"solver={solver}", "device=cpu",
                                                       "seanet.n_filters=8", "gaussian_splat.n_iters=2"])
    model = builders.get_compression_model(cfg)
    assert model.frame_rate == 75 and model.quantizer.dimension == 32
    x = torch.randn(1, 1, 72000) * 0.1
    res = model(x)
    assert res.x.shape == x.shape and res.penalty is not None


def test_unsupported_option_rejected():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["solver=compression/gaussian_24khz", "device=cpu",
                                                       "+gaussian_splat.basis_type=gabor"])
    with pytest.raises(ValueError):
        builders.get_compression_model(cfg)
