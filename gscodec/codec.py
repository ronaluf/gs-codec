# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""GS-Codec inference API."""

import contextlib
import copy
import json
import math
import struct
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from . import ptq
from .data.audio_utils import convert_audio
from .models.encodec import EncodecModel
from .modules import SEANetDecoder, SEANetEncoder
from .quantization import GaussianSplatQuantizer

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
CODEBOOKS_NAME = "codebooks.json"
MODES = ("predictor", "iterative")

_MAGIC = b"GSC1"
_HEADER = struct.Struct("<4sBBBBHIII")


@dataclass
class GSCode:
    """Quantized Gaussian-splatting parameters of a waveform, one row per segment.

    Attributes:
        scales: Scale indices, ``(S, N_G)``.
        weights: Amplitude indices, ``(S, C, N_G)``.
        positions: Center indices ``(S, N_G)``, or ``None`` for grid centers.
        n_bits: Bits per scale and amplitude.
        position_bits: Bits per center (0 for grid centers).
        mode: Encoder that produced the parameters (``"predictor"`` or ``"iterative"``).
        length: Waveform length in samples.
        sample_rate: Sample rate of the codec.
        segment_duration: Segment duration in seconds.
    """
    scales: torch.Tensor
    weights: torch.Tensor
    positions: tp.Optional[torch.Tensor]
    n_bits: int
    position_bits: int
    mode: str
    length: int
    sample_rate: int
    segment_duration: float

    @property
    def n_segments(self) -> int:
        return self.scales.shape[0]

    @property
    def n_gaussians(self) -> int:
        return self.scales.shape[-1]

    @property
    def dimension(self) -> int:
        return self.weights.shape[1]

    @property
    def num_bits(self) -> int:
        """Payload size in bits."""
        return self.n_segments * ptq.segment_bits(self.n_gaussians, self.dimension, self.n_bits, self.position_bits)

    @property
    def kbps(self) -> float:
        """Payload bitrate in kbit/s."""
        return self.num_bits / (self.n_segments * self.segment_duration) / 1000

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        flags = int(self.positions is not None)
        header = _HEADER.pack(_MAGIC, flags, self.n_bits, self.position_bits, MODES.index(self.mode),
                              self.n_gaussians, self.n_segments, self.length, self.sample_rate)
        dims = struct.pack("<Hf", self.dimension, self.segment_duration)
        body = [ptq.pack_bits(self.scales.cpu().numpy(), self.n_bits),
                ptq.pack_bits(self.weights.cpu().numpy(), self.n_bits)]
        if self.positions is not None:
            body.append(ptq.pack_bits(self.positions.cpu().numpy(), self.position_bits))
        return header + dims + b"".join(body)

    @classmethod
    def from_bytes(cls, data: bytes) -> "GSCode":
        magic, flags, n_bits, position_bits, mode, N, S, length, sr = _HEADER.unpack_from(data)
        if magic != _MAGIC:
            raise ValueError("Not a GS-Codec bitstream.")
        C, seg_dur = struct.unpack_from("<Hf", data, _HEADER.size)
        offset = _HEADER.size + struct.calcsize("<Hf")

        def take(count: int, bits: int) -> torch.Tensor:
            nonlocal offset
            nbytes = (count * bits + 7) // 8
            values = ptq.unpack_bits(data[offset:offset + nbytes], count, bits)
            offset += nbytes
            return torch.from_numpy(values)

        scales = take(S * N, n_bits).view(S, N)
        weights = take(S * C * N, n_bits).view(S, C, N)
        positions = take(S * N, position_bits).view(S, N) if flags & 1 else None
        return cls(scales, weights, positions, n_bits, position_bits, MODES[mode], length, sr, seg_dur)


def build_model(config: tp.Dict[str, tp.Any]) -> EncodecModel:
    """Build the model from a ``config.json`` dict."""
    seanet = copy.deepcopy(config["seanet"])
    encoder_override = seanet.pop("encoder", {})
    decoder_override = seanet.pop("decoder", {})
    encoder_kwargs = {**seanet, **encoder_override}
    decoder_kwargs = {**seanet, **decoder_override}
    encoder = SEANetEncoder(**encoder_kwargs)
    decoder = SEANetDecoder(**decoder_kwargs)
    quantizer = GaussianSplatQuantizer(dimension=encoder.dimension, **config["gaussian_splat"])
    sample_rate = int(config["sample_rate"])
    frame_rate = sample_rate // int(encoder.hop_length)
    model = EncodecModel(encoder, decoder, quantizer, frame_rate=frame_rate, sample_rate=sample_rate,
                         channels=int(config.get("channels", 1)))
    if quantizer.amortized_predictor_enabled:
        quantizer.build_predictor(int(round(config["segment_duration"] * frame_rate)))
    return model


class GSCodec(nn.Module):
    """Neural audio codec with a Gaussian-splatting bottleneck.

    Example:
        >>> codec = GSCodec.from_pretrained("ronaluf/gs-codec-24khz")
        >>> code = codec.encode(wav, sample_rate=sr)
        >>> print(f"{code.kbps:.2f} kbps")
        >>> wav_hat = codec.decode(code)
    """

    def __init__(self, model: EncodecModel, config: tp.Dict[str, tp.Any],
                 codebooks: tp.Optional[tp.Dict[str, ptq.Codebook]] = None):
        super().__init__()
        self.model = model.eval()
        self.config = config
        self.codebooks: tp.Dict[str, ptq.Codebook] = dict(codebooks or {})
        if self.quantizer.normalize_input:
            raise ValueError("GSCodec expects gaussian_splat.normalize_input=false.")

    @classmethod
    def from_pretrained(cls, name_or_path: tp.Union[str, Path], device: tp.Union[str, torch.device] = "cpu",
                        revision: tp.Optional[str] = None) -> "GSCodec":
        """Load from a local directory or a Hugging Face Hub repository."""
        path = Path(name_or_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(str(name_or_path), revision=revision))
        from safetensors.torch import load_file
        config = json.loads((path / CONFIG_NAME).read_text())
        model = build_model(config)
        model.load_state_dict(load_file(str(path / WEIGHTS_NAME)), strict=True)
        codebooks = {}
        if (path / CODEBOOKS_NAME).exists():
            raw = json.loads((path / CODEBOOKS_NAME).read_text())
            codebooks = {k: ptq.Codebook.from_dict(v) for k, v in raw.items()}
        return cls(model, config, codebooks).to(device)

    def save_pretrained(self, path: tp.Union[str, Path]) -> None:
        """Write ``config.json``, ``model.safetensors`` and ``codebooks.json`` to ``path``."""
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / CONFIG_NAME).write_text(json.dumps(self.config, indent=2) + "\n")
        state = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()}
        save_file(state, str(path / WEIGHTS_NAME), metadata={"format": "pt"})
        if self.codebooks:
            raw = {k: cb.to_dict() for k, cb in sorted(self.codebooks.items())}
            (path / CODEBOOKS_NAME).write_text(json.dumps(raw, indent=1) + "\n")

    @property
    def quantizer(self) -> GaussianSplatQuantizer:
        return self.model.quantizer

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def sample_rate(self) -> int:
        return int(self.model.sample_rate)

    @property
    def frame_rate(self) -> int:
        return int(self.model.frame_rate)

    @property
    def segment_duration(self) -> float:
        return float(self.config["segment_duration"])

    @property
    def segment_length(self) -> int:
        return int(round(self.segment_duration * self.sample_rate))

    @property
    def dimension(self) -> int:
        return int(self.quantizer.dimension)

    @property
    def has_predictor(self) -> bool:
        return self.quantizer.predictor is not None

    def _defaults(self, **kwargs) -> tp.Dict[str, tp.Any]:
        out = dict(self.config.get("defaults", {}))
        out.update({k: v for k, v in kwargs.items() if v is not None})
        if out.get("mode") not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {out.get('mode')!r}.")
        if out["mode"] == "predictor" and not self.has_predictor:
            raise ValueError("This checkpoint has no GS Predictor, use mode='iterative'.")
        return out

    @contextlib.contextmanager
    def _iterative_settings(self, n_gaussians: int, freeze_positions: bool):
        q = self.quantizer
        overrides = {"n_gaussians": n_gaussians, "freeze_positions": freeze_positions,
                     **self.config.get("iterative", {})}
        saved = {k: getattr(q, k) for k in overrides}
        try:
            for k, v in overrides.items():
                setattr(q, k, v)
            yield
        finally:
            for k, v in saved.items():
                setattr(q, k, v)

    def fit(self, z: torch.Tensor, mode: str, n_gaussians: int,
            freeze_positions: bool) -> tp.Dict[str, torch.Tensor]:
        """Primitive parameters for latents ``z`` of shape ``(B, C, T)``."""
        B, C, T = z.shape
        if mode == "predictor":
            with torch.no_grad():
                pred = self.quantizer.predict_params_amortized(z, T, n_gaussians=n_gaussians,
                                                               frozen_mode=freeze_positions)
            params = {"means": pred["pos"], "scales": pred["scales"], "weights": pred["weights"]}
        else:
            with self._iterative_settings(n_gaussians, freeze_positions), torch.enable_grad():
                p = self.quantizer._init_params(B, C, T, z.device, target=z.detach())
                p = self.quantizer._optimize(p, z.detach(), T)
            params = {"means": p["means"], "scales": p["scales"], "weights": p["weights"]}
        if freeze_positions:
            params["means"] = self._grid(B, n_gaussians, T, z.device)
        return params

    def render(self, params: tp.Dict[str, torch.Tensor], T: int) -> torch.Tensor:
        B, C = params["weights"].shape[:2]
        full = {**params, "bias": torch.zeros(B, C, device=params["weights"].device)}
        return self.quantizer.render(full, T)

    @staticmethod
    def _grid(B: int, N: int, T: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(0.0, float(T - 1), N, device=device).view(1, 1, N).expand(B, 1, N).contiguous()

    def _segments(self, wav: torch.Tensor, sample_rate: tp.Optional[int]) -> tp.Tuple[torch.Tensor, int]:
        if wav.dim() == 1:
            wav = wav[None]
        if wav.dim() != 2:
            raise ValueError("Expected a waveform of shape (samples,) or (channels, samples).")
        wav = convert_audio(wav.float(), sample_rate or self.sample_rate, self.sample_rate, 1)
        length = wav.shape[-1]
        L = self.segment_length
        n_seg = max(1, math.ceil(length / L))
        wav = torch.nn.functional.pad(wav, (0, n_seg * L - length))
        return wav.view(1, n_seg, L).transpose(0, 1).contiguous(), length

    def _codebook(self, mode: str, freeze_positions: bool, n_gaussians: int, n_bits: int) -> ptq.Codebook:
        key = ptq.codebook_key(mode, freeze_positions, n_gaussians, n_bits)
        if key not in self.codebooks:
            raise KeyError(f"No codebook '{key}'. Available: {sorted(self.codebooks)}. "
                           "Fit one with scripts/calibrate_codebooks.py.")
        return self.codebooks[key]

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sample_rate: tp.Optional[int] = None,
               n_gaussians: tp.Optional[int] = None, n_bits: tp.Optional[int] = None,
               mode: tp.Optional[str] = None, freeze_positions: tp.Optional[bool] = None,
               batch_size: int = 16) -> GSCode:
        """Encode a waveform ``(samples,)`` or ``(channels, samples)`` to a :class:`GSCode`."""
        opts = self._defaults(n_gaussians=n_gaussians, n_bits=n_bits, mode=mode,
                              freeze_positions=freeze_positions)
        N, B, mode, frozen = opts["n_gaussians"], opts["n_bits"], opts["mode"], opts["freeze_positions"]
        position_bits = 0 if frozen else int(opts.get("position_bits", 10))
        cb = self._codebook(mode, frozen, N, B)
        segs, length = self._segments(wav, sample_rate)
        sc_idx, w_idx, pos_idx = [], [], []
        for chunk in segs.split(batch_size):
            z = self.model.encoder(chunk.to(self.device))
            T = z.shape[-1]
            params = self.fit(z, mode, N, frozen)
            sc_idx.append(ptq.quantize_codebook(params["scales"][:, 0], cb.scales).cpu())
            w_idx.append(ptq.quantize_codebook(params["weights"], cb.weights).cpu())
            if not frozen:
                pos_idx.append(ptq.quantize_uniform(params["means"][:, 0], 0.0, float(T - 1), position_bits).cpu())
        return GSCode(torch.cat(sc_idx), torch.cat(w_idx), torch.cat(pos_idx) if pos_idx else None,
                      B, position_bits, mode, length, self.sample_rate, self.segment_duration)

    @torch.no_grad()
    def decode(self, code: GSCode, batch_size: int = 16) -> torch.Tensor:
        """Decode a :class:`GSCode` to a waveform of shape ``(1, samples)`` at ``sample_rate``."""
        cb = self._codebook(code.mode, code.positions is None, code.n_gaussians, code.n_bits)
        T = int(round(code.segment_duration * self.frame_rate))
        out = []
        for s in range(0, code.n_segments, batch_size):
            sl = slice(s, s + batch_size)
            scales = ptq.dequantize_codebook(code.scales[sl].to(self.device), cb.scales)[:, None]
            weights = ptq.dequantize_codebook(code.weights[sl].to(self.device), cb.weights)
            if code.positions is None:
                means = self._grid(scales.shape[0], code.n_gaussians, T, self.device)
            else:
                means = ptq.dequantize_uniform(code.positions[sl].to(self.device), 0.0, float(T - 1),
                                               code.position_bits)[:, None]
            z_hat = self.render({"means": means, "scales": scales, "weights": weights}, T)
            y = self.model.decoder(z_hat)[..., :self.segment_length]
            out.append(y[:, 0].cpu())
        return torch.cat(out).reshape(1, -1)[:, :code.length]

    def reconstruct(self, wav: torch.Tensor, sample_rate: tp.Optional[int] = None, **kwargs) -> torch.Tensor:
        """Encode and decode; returns ``(1, samples)`` at ``sample_rate``."""
        return self.decode(self.encode(wav, sample_rate=sample_rate, **kwargs))
