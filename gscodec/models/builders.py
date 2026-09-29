# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE.audiocraft and LICENSE files in the root directory of this source tree.
"""Build the compression model and its components from a Hydra config."""

import inspect

import omegaconf

from .. import quantization as qt
from ..modules import SEANetDecoder, SEANetEncoder
from ..utils.utils import dict_from_config
from .encodec import CompressionModel, EncodecModel


def get_quantizer(quantizer: str, cfg: omegaconf.DictConfig, dimension: int) -> qt.BaseQuantizer:
    if quantizer == "no_quant":
        return qt.DummyQuantizer()
    if quantizer != "gaussian_splat":
        raise KeyError(f"Unexpected quantizer {quantizer}")
    kwargs = dict_from_config(getattr(cfg, quantizer))
    valid = set(inspect.signature(qt.GaussianSplatQuantizer.__init__).parameters) - {"self", "dimension"}
    unknown = sorted(set(kwargs) - valid)
    if unknown:
        raise ValueError(f"Unknown gaussian_splat options: {unknown}")
    kwargs["dimension"] = dimension
    return qt.GaussianSplatQuantizer(**kwargs)


def get_encodec_autoencoder(encoder_name: str, cfg: omegaconf.DictConfig):
    if encoder_name != "seanet":
        raise KeyError(f"Unexpected autoencoder {encoder_name}")
    kwargs = dict_from_config(getattr(cfg, "seanet"))
    encoder_override_kwargs = kwargs.pop("encoder")
    decoder_override_kwargs = kwargs.pop("decoder")
    encoder = SEANetEncoder(**{**kwargs, **encoder_override_kwargs})
    decoder = SEANetDecoder(**{**kwargs, **decoder_override_kwargs})
    return encoder, decoder


def get_compression_model(cfg: omegaconf.DictConfig) -> CompressionModel:
    """Instantiate a compression model."""
    if cfg.compression_model != "encodec":
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")
    kwargs = dict_from_config(getattr(cfg, "encodec"))
    encoder_name = kwargs.pop("autoencoder")
    quantizer_name = kwargs.pop("quantizer")
    encoder, decoder = get_encodec_autoencoder(encoder_name, cfg)
    quantizer = get_quantizer(quantizer_name, cfg, encoder.dimension)
    frame_rate = kwargs["sample_rate"] // encoder.hop_length
    renormalize = kwargs.pop("renormalize", False)
    kwargs.pop("renorm", None)
    return EncodecModel(encoder, decoder, quantizer, frame_rate=frame_rate,
                        renormalize=renormalize, **kwargs).to(cfg.device)
