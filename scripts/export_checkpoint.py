#!/usr/bin/env python3
# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Export a training checkpoint to a pretrained model directory.

The output directory holds ``config.json``, ``model.safetensors`` and, optionally,
``codebooks.json``, and loads with ``GSCodec.from_pretrained(out_dir)``.

Example:
    python scripts/export_checkpoint.py outputs/xps/<sig>/checkpoint.th pretrained/gs-codec-24khz \
        --codebook predictor/free/ng102/b5=codebooks/ng102_b5.json
"""

import argparse
import inspect
import json
from pathlib import Path

import omegaconf
import torch

from gscodec import GSCodec, __version__, build_model, ptq
from gscodec.modules import SEANetDecoder, SEANetEncoder
from gscodec.quantization import GaussianSplatQuantizer

ITERATIVE_EVAL = {
    "n_iters": 300,
    "lr": 0.05,
    "lr_position_mult": 2.0,
    "lr_schedule": "cosine",
    "lr_final_ratio": 0.01,
    "lr_warmup_iters": 5,
    "param_clamp_scale_min": 0.1,
    "param_clamp_scale_max": 100.0,
    "param_clamp_weight_min": -3.0,
    "param_clamp_weight_max": 3.0,
}
TRAINING_ONLY = ("warmup_mode", "amortized_pretrained_codec_path", "amortized_param_noise",
                 "amortized_pos_lr_mult", "amortized_frozen_mode_prob")


def load_training_checkpoint(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("xp.cfg")
    hydra_cfg = path.parent / ".hydra" / "config.yaml"
    if cfg is None and hydra_cfg.exists():
        cfg = omegaconf.OmegaConf.load(hydra_cfg)
    if cfg is None:
        raise ValueError(f"No configuration found in {path}.")
    weights = state["best_state"]["model"] if "best_state" in state else state["model"]
    return cfg, weights


def _accepted(cls) -> set:
    return set(inspect.signature(cls.__init__).parameters) - {"self"}


def make_config(cfg: omegaconf.DictConfig, weights: dict) -> dict:
    seanet = omegaconf.OmegaConf.to_container(cfg.seanet, resolve=True)
    enc = {k: v for k, v in seanet.pop("encoder", {}).items() if k in _accepted(SEANetEncoder)}
    dec = {k: v for k, v in seanet.pop("decoder", {}).items() if k in _accepted(SEANetDecoder)}
    seanet = {k: v for k, v in seanet.items() if k in _accepted(SEANetEncoder) & _accepted(SEANetDecoder)}
    seanet.update(encoder=enc, decoder=dec)
    gs = omegaconf.OmegaConf.to_container(cfg.gaussian_splat, resolve=True)
    if gs.get("basis_type", "gaussian") != "gaussian":
        raise ValueError(f"Unsupported basis_type: {gs['basis_type']}")
    valid = _accepted(GaussianSplatQuantizer) - {"dimension"}
    gs = {k: v for k, v in gs.items() if k in valid and k not in TRAINING_ONLY}
    has_predictor = any(".predictor." in k for k in weights)
    gs["amortized_predictor"] = has_predictor
    return {
        "model_type": "gs-codec",
        "gscodec_version": __version__,
        "sample_rate": int(cfg.sample_rate),
        "channels": int(cfg.channels),
        "segment_duration": float(cfg.dataset.segment_duration),
        "seanet": seanet,
        "gaussian_splat": gs,
        "iterative": dict(ITERATIVE_EVAL),
        "defaults": {
            "mode": "predictor" if has_predictor else "iterative",
            "n_gaussians": 102,
            "n_bits": 5,
            "freeze_positions": False,
            "position_bits": 10,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Training checkpoint (.th).")
    parser.add_argument("out_dir", type=Path, help="Output model directory.")
    parser.add_argument("--codebook", action="append", default=[], metavar="KEY=PATH",
                        help="Add a codebook JSON with 'scales' and 'weights' lists under KEY "
                             "(mode/grid|free/ng<N>/b<B>). Repeatable.")
    args = parser.parse_args()

    cfg, weights = load_training_checkpoint(args.checkpoint)
    config = make_config(cfg, weights)
    model = build_model(config)
    model.load_state_dict(weights, strict=True)

    codebooks = {}
    for item in args.codebook:
        key, path = item.split("=", 1)
        codebooks[key] = ptq.Codebook.from_dict(json.loads(Path(path).read_text()))
    GSCodec(model, config, codebooks).save_pretrained(args.out_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Exported {n_params / 1e6:.2f}M parameters to {args.out_dir} "
          f"({'with' if config['gaussian_splat']['amortized_predictor'] else 'without'} GS Predictor, "
          f"{len(codebooks)} codebooks).")


if __name__ == "__main__":
    main()
