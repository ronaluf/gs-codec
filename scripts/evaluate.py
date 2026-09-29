#!/usr/bin/env python3
# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compute speech-quality metrics on the ``ref/`` and ``deg/`` folders written by ``reconstruct.py``.

Metrics (16 kHz): PESQ-WB, PESQ-NB, STOI, UTMOS (utmos22_strong), corpus WER with
``facebook/hubert-large-ls960-ft`` (requires LibriSpeech-style ``*.trans.txt`` transcripts),
and speaker similarity with WavLM-Large + ECAPA-TDNN (requires ``stopes`` and the checkpoint).

Example:
    python scripts/evaluate.py runs/pred_102_b5 --transcripts LibriSpeech/test-clean
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

SR = 16000


def load(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr} Hz.")
    return wav if wav.ndim == 1 else wav.mean(axis=1)


def normalize_text(text: str) -> str:
    text = re.sub(r"[^a-z\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def read_transcripts(root: Path) -> dict:
    out = {}
    for f in root.rglob("*.trans.txt"):
        for line in f.read_text().splitlines():
            utt, _, text = line.partition(" ")
            out[utt] = text
    return out


def pesq_stoi(pairs):
    from pesq import pesq, NoUtterancesError
    from pystoi import stoi
    rows = {"pesq_wb": [], "pesq_nb": [], "stoi": []}
    for ref_p, deg_p in tqdm(pairs, desc="PESQ/STOI"):
        ref, deg = load(ref_p), load(deg_p)
        n = min(len(ref), len(deg))
        ref, deg = ref[:n], deg[:n]
        try:
            rows["pesq_wb"].append(pesq(SR, ref, deg, "wb"))
            rows["pesq_nb"].append(pesq(SR, ref, deg, "nb"))
        except NoUtterancesError:
            pass
        rows["stoi"].append(stoi(ref, deg, SR, extended=False))
    return rows


def utmos(pairs, device):
    from gscodec.metrics.utmos import UTMOSScore
    scorer = UTMOSScore(device)
    return {"utmos": [float(scorer.score(torch.from_numpy(load(d)))[0]) for _, d in tqdm(pairs, desc="UTMOS")]}


@torch.no_grad()
def wer(pairs, transcripts, device):
    from jiwer import wer as jiwer_wer
    from transformers import HubertForCTC, Wav2Vec2Processor
    name = "facebook/hubert-large-ls960-ft"
    processor = Wav2Vec2Processor.from_pretrained(name)
    model = HubertForCTC.from_pretrained(name).to(device).eval()
    refs, hyps = [], []
    for _, deg_p in tqdm(pairs, desc="WER"):
        if deg_p.stem not in transcripts:
            continue
        inputs = processor(load(deg_p), sampling_rate=SR, return_tensors="pt").to(device)
        ids = model(inputs.input_values).logits.argmax(dim=-1)
        refs.append(normalize_text(transcripts[deg_p.stem]))
        hyps.append(normalize_text(processor.batch_decode(ids)[0]))
    return {"wer": [jiwer_wer(r, h) * 100 for r, h in zip(refs, hyps)]}, jiwer_wer(refs, hyps) * 100


def speaker_similarity(pairs, checkpoint):
    from stopes.eval.vocal_style_similarity.vocal_style_sim_tool import compute_cosine_similarity, get_embedder
    embedder = get_embedder(model_name="valle", model_path=checkpoint)
    ref = embedder([str(r) for r, _ in pairs])
    deg = embedder([str(d) for _, d in pairs])
    return {"sim": [float(s) for s in compute_cosine_similarity(ref, deg)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Directory with ref/ and deg/ subfolders.")
    parser.add_argument("--transcripts", type=Path, default=None, help="Root with *.trans.txt files (enables WER).")
    parser.add_argument("--sim_checkpoint", default=None, help="WavLM-Large ECAPA-TDNN checkpoint (enables SIM).")
    parser.add_argument("--no_utmos", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ref_dir, deg_dir = args.run_dir / "ref", args.run_dir / "deg"
    stems = sorted({p.stem for p in ref_dir.glob("*.wav")} & {p.stem for p in deg_dir.glob("*.wav")})
    pairs = [(ref_dir / f"{s}.wav", deg_dir / f"{s}.wav") for s in stems]
    if not pairs:
        raise SystemExit(f"No ref/deg pairs in {args.run_dir}.")

    rows = pesq_stoi(pairs)
    if not args.no_utmos:
        rows.update(utmos(pairs, args.device))
    corpus_wer = None
    if args.transcripts is not None:
        per_file_wer, corpus_wer = wer(pairs, read_transcripts(args.transcripts), args.device)
        rows.update(per_file_wer)
    if args.sim_checkpoint is not None:
        rows.update(speaker_similarity(pairs, args.sim_checkpoint))

    results = {"n_files": len(pairs)}
    results.update({k: float(np.mean(v)) for k, v in rows.items() if v})
    if corpus_wer is not None:
        results["wer"] = float(corpus_wer)
    (args.run_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.run_dir / "metrics_per_file.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
