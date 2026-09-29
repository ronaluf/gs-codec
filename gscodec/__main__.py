# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Command-line interface.

    python -m gscodec encode input.wav output.gsc [--n_gaussians 102 --n_bits 5]
    python -m gscodec decode input.gsc output.wav
    python -m gscodec reconstruct input.wav output.wav
    python -m gscodec bitrate --n_gaussians 102 --n_bits 5
"""

import argparse
from pathlib import Path

DEFAULT_MODEL = "ronaluf/gs-codec-24khz"


def _add_codec_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model directory or Hugging Face repository.")
    p.add_argument("--mode", choices=["predictor", "iterative"], default=None)
    p.add_argument("--n_gaussians", type=int, default=None)
    p.add_argument("--n_bits", type=int, default=None)
    p.add_argument("--freeze_positions", action="store_true", help="Centers on the uniform grid (not transmitted).")
    p.add_argument("--device", default=None)


def _load(args):
    import torch
    from . import GSCodec
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return GSCodec.from_pretrained(args.model, device=device)


def _encode_kwargs(args):
    return dict(n_gaussians=args.n_gaussians, n_bits=args.n_bits, mode=args.mode,
                freeze_positions=args.freeze_positions or None)


def main() -> None:
    parser = argparse.ArgumentParser(prog="gscodec", description="GS-Codec neural audio codec.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("encode", "reconstruct"):
        p = sub.add_parser(name)
        p.add_argument("input", type=Path)
        p.add_argument("output", type=Path)
        _add_codec_args(p)
    p = sub.add_parser("decode")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None)
    p = sub.add_parser("bitrate", help="Bitrate of an operating point.")
    p.add_argument("--n_gaussians", type=int, required=True)
    p.add_argument("--n_bits", type=int, required=True)
    p.add_argument("--dimension", type=int, default=32)
    p.add_argument("--position_bits", type=int, default=10)
    p.add_argument("--freeze_positions", action="store_true")
    p.add_argument("--segment_duration", type=float, default=3.0)
    args = parser.parse_args()

    if args.command == "bitrate":
        from .ptq import segment_bits
        pos_bits = 0 if args.freeze_positions else args.position_bits
        bits = segment_bits(args.n_gaussians, args.dimension, args.n_bits, pos_bits)
        print(f"{bits} bits / {args.segment_duration:g} s = {bits / args.segment_duration / 1000:.3f} kbps")
        return

    from .codec import GSCode
    from .data.audio import audio_read, audio_write
    codec = _load(args)
    if args.command == "decode":
        wav = codec.decode(GSCode.from_bytes(args.input.read_bytes()))
    else:
        wav, sr = audio_read(args.input)
        code = codec.encode(wav, sample_rate=sr, **_encode_kwargs(args))
        print(f"{code.mode}, N_G={code.n_gaussians}, {code.n_bits}-bit, {code.kbps:.3f} kbps")
        if args.command == "encode":
            args.output.write_bytes(code.to_bytes())
            return
        wav = codec.decode(code)
    audio_write(args.output.with_suffix(""), wav, codec.sample_rate, format=args.output.suffix.lstrip(".") or "wav",
                strategy="clip")


if __name__ == "__main__":
    main()
