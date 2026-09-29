#!/usr/bin/env python3
# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Encode a set of files to bitstreams, decode them from disk, and write reference/decoded audio.

Output layout::

    out_dir/
      bitstreams/<stem>.gsc   # serialized GSCode
      ref/<stem>.wav          # reference at --eval_sample_rate
      deg/<stem>.wav          # decoded at --eval_sample_rate
      summary.json            # operating point and measured bitrate

Example:
    python scripts/reconstruct.py pretrained/gs-codec-24khz --files test_clean.txt --out_dir runs/pred_102_b5 \
        --mode predictor --n_gaussians 102 --n_bits 5
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from gscodec import GSCode, GSCodec
from gscodec.data.audio import audio_read
from gscodec.data.audio_utils import convert_audio

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def list_audio(spec: str):
    p = Path(spec)
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
    return [Path(line.strip()) for line in p.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Pretrained model directory or Hugging Face repository.")
    parser.add_argument("--files", required=True, help="Audio directory or text file with one path per line.")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["predictor", "iterative"], default=None)
    parser.add_argument("--n_gaussians", type=int, default=None)
    parser.add_argument("--n_bits", type=int, default=None)
    parser.add_argument("--freeze_positions", action="store_true", help="Centers on the uniform grid.")
    parser.add_argument("--min_duration", type=float, default=0.0)
    parser.add_argument("--max_duration", type=float, default=float("inf"))
    parser.add_argument("--eval_sample_rate", type=int, default=16000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    codec = GSCodec.from_pretrained(args.model, device=args.device)
    for sub in ("bitstreams", "ref", "deg"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    total_bits, total_seconds, n_files, last = 0, 0.0, 0, None
    for f in tqdm(list_audio(args.files)):
        wav, sr = audio_read(f)
        duration = wav.shape[-1] / sr
        if not args.min_duration <= duration <= args.max_duration:
            continue
        code = codec.encode(wav, sample_rate=sr, n_gaussians=args.n_gaussians, n_bits=args.n_bits,
                            mode=args.mode, freeze_positions=args.freeze_positions or None)
        path = args.out_dir / "bitstreams" / f"{f.stem}.gsc"
        path.write_bytes(code.to_bytes())
        decoded = codec.decode(GSCode.from_bytes(path.read_bytes()))

        ref = convert_audio(wav.float(), sr, args.eval_sample_rate, 1)
        deg = convert_audio(decoded, codec.sample_rate, args.eval_sample_rate, 1)[..., :ref.shape[-1]]
        sf.write(args.out_dir / "ref" / f"{f.stem}.wav", ref[0].numpy(), args.eval_sample_rate)
        sf.write(args.out_dir / "deg" / f"{f.stem}.wav", deg[0].numpy(), args.eval_sample_rate)
        total_bits += code.num_bits
        total_seconds += code.n_segments * code.segment_duration
        n_files += 1
        last = code

    if last is None:
        raise SystemExit("No files matched.")
    summary = {
        "model": str(args.model),
        "mode": last.mode,
        "n_gaussians": last.n_gaussians,
        "n_bits": last.n_bits,
        "position_bits": last.position_bits,
        "kbps": total_bits / total_seconds / 1000,
        "n_files": n_files,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
