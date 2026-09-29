#!/usr/bin/env python3
# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fit k-means codebooks for scales and amplitudes on a calibration set.

Codebooks are stored in ``<model_dir>/codebooks.json`` under keys
``<mode>/<grid|free>/ng<N_G>/b<B>``. Use a held-out set (e.g. LibriTTS dev-clean).

Example:
    python scripts/calibrate_codebooks.py pretrained/gs-codec-24khz --files dev_clean.txt \
        --mode predictor --n_gaussians 102 --n_bits 4 5 6
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from gscodec import GSCodec, ptq
from gscodec.codec import CODEBOOKS_NAME
from gscodec.data.audio import audio_read

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def list_audio(spec: str):
    p = Path(spec)
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
    return [Path(line.strip()) for line in p.read_text().splitlines() if line.strip()]


@torch.no_grad()
def collect_pool(codec: GSCodec, files, mode: str, n_gaussians: int, freeze_positions: bool,
                 segments_per_file: int, batch_size: int):
    scales, weights, batch = [], [], []

    def flush():
        if not batch:
            return
        z = codec.model.encoder(torch.stack(batch).to(codec.device))
        params = codec.fit(z, mode, n_gaussians, freeze_positions)
        scales.append(params["scales"].flatten().cpu())
        weights.append(params["weights"].flatten().cpu())
        batch.clear()

    for f in tqdm(files, desc=f"{mode} N_G={n_gaussians}"):
        wav, sr = audio_read(f)
        segs, _ = codec._segments(wav, sr)
        batch.extend(segs[:segments_per_file])
        if len(batch) >= batch_size:
            flush()
    flush()
    return torch.cat(scales), torch.cat(weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Pretrained model directory.")
    parser.add_argument("--files", required=True, help="Audio directory or text file with one path per line.")
    parser.add_argument("--mode", choices=["predictor", "iterative"], default="predictor")
    parser.add_argument("--n_gaussians", type=int, nargs="+", default=[102])
    parser.add_argument("--n_bits", type=int, nargs="+", default=[5])
    parser.add_argument("--freeze_positions", action="store_true", help="Centers on the uniform grid.")
    parser.add_argument("--max_files", type=int, default=200)
    parser.add_argument("--segments_per_file", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    codec = GSCodec.from_pretrained(args.model, device=args.device)
    files = list_audio(args.files)[:args.max_files]
    for ng in args.n_gaussians:
        pool = collect_pool(codec, files, args.mode, ng, args.freeze_positions,
                            args.segments_per_file, args.batch_size)
        for nb in args.n_bits:
            key = ptq.codebook_key(args.mode, args.freeze_positions, ng, nb)
            codec.codebooks[key] = ptq.Codebook.fit(*pool, n_bits=nb)
            print(f"{key}: fitted on {pool[0].numel()} scales / {pool[1].numel()} amplitudes")

    out = Path(args.model) / CODEBOOKS_NAME
    raw = {k: cb.to_dict() for k, cb in sorted(codec.codebooks.items())}
    out.write_text(json.dumps(raw, indent=1) + "\n")
    print(f"Wrote {len(raw)} codebooks to {out}")


if __name__ == "__main__":
    main()
