# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""UTMOS (utmos22_strong) automatic MOS prediction."""

import torch


class UTMOSScore:
    """UTMOS22 strong learner via ``tarepan/SpeechMOS``. Expects 16 kHz audio."""

    SAMPLE_RATE = 16000

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                                    trust_repo=True).to(device).eval()

    @torch.no_grad()
    def score(self, wavs: torch.Tensor) -> torch.Tensor:
        """Score ``(T,)`` or ``(B, T)`` waveforms; returns ``(B,)`` MOS estimates."""
        if wavs.dim() == 1:
            wavs = wavs[None]
        return self.model(wavs.to(self.device), self.SAMPLE_RATE)
