# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE.audiocraft file in the root directory of this source tree.
"""Base classes for the datasets that also provide non-audio metadata,
e.g. description, text transcription etc.
"""
from dataclasses import dataclass
import logging
import typing as tp

import torch

from .audio_dataset import AudioDataset, AudioMeta, SegmentInfo
from ..environment import Environment


logger = logging.getLogger(__name__)


def _clusterify_meta(meta: AudioMeta) -> AudioMeta:
    """Monkey-patch meta to match cluster specificities."""
    meta.path = Environment.apply_dataset_mappers(meta.path)
    if meta.info_path is not None:
        meta.info_path.zip_path = Environment.apply_dataset_mappers(meta.info_path.zip_path)
    return meta


def clusterify_all_meta(meta: tp.List[AudioMeta]) -> tp.List[AudioMeta]:
    """Monkey-patch all meta to match cluster specificities."""
    return [_clusterify_meta(m) for m in meta]


@dataclass
class AudioInfo(SegmentInfo):
    """Segment metadata returned along with the waveform."""
    audio_tokens: tp.Optional[torch.Tensor] = None


class InfoAudioDataset(AudioDataset):
    """AudioDataset that always returns metadata as AudioInfo along with the audio waveform.

    See `gscodec.data.audio_dataset.AudioDataset` for initialization arguments.
    """
    def __init__(self, meta: tp.List[AudioMeta], **kwargs):
        super().__init__(clusterify_all_meta(meta), **kwargs)

    def __getitem__(self, index: int) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, AudioInfo]]:
        if not self.return_info:
            wav = super().__getitem__(index)
            assert isinstance(wav, torch.Tensor)
            return wav
        wav, meta = super().__getitem__(index)
        return wav, AudioInfo(**meta.to_dict())
