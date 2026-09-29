# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Encode a file to a GS-Codec bitstream and decode it back."""
import sys

from gscodec import GSCode, GSCodec
from gscodec.data.audio import audio_read, audio_write

codec = GSCodec.from_pretrained("ronaluf/gs-codec-24khz")
wav, sr = audio_read(sys.argv[1] if len(sys.argv) > 1 else "input.wav")

code = codec.encode(wav, sample_rate=sr, n_gaussians=102, n_bits=5)  # GS Predictor, ~5.95 kbps
print(f"{code.n_segments} segments, {code.kbps:.2f} kbps, {len(code.to_bytes())} bytes")

wav_hat = codec.decode(GSCode.from_bytes(code.to_bytes()))
audio_write("reconstruction", wav_hat, codec.sample_rate)
