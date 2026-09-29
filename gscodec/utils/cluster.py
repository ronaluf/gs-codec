# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE.audiocraft file in the root directory of this source tree.

"""
Utility functions for SLURM configuration and cluster settings.
"""

from enum import Enum
import os
import typing as tp

import omegaconf


class ClusterType(Enum):
    LOCAL_DARWIN = "darwin"
    DEFAULT = "default"  # any other machine or cluster.


def _guess_cluster_type() -> ClusterType:
    if os.uname().sysname == "Darwin":
        return ClusterType.LOCAL_DARWIN
    return ClusterType.DEFAULT


def get_cluster_type(
    cluster_type: tp.Optional[ClusterType] = None,
) -> tp.Optional[ClusterType]:
    if cluster_type is None:
        return _guess_cluster_type()

    return cluster_type


def get_slurm_parameters(
    cfg: omegaconf.DictConfig, cluster_type: tp.Optional[ClusterType] = None
) -> omegaconf.DictConfig:
    """Update SLURM parameters in configuration based on cluster type.
    If the cluster type is not specify, it infers it automatically.
    """
    from ..environment import Environment
    slurm_exclude = Environment.get_slurm_exclude()
    if slurm_exclude is not None:
        cfg["exclude"] = slurm_exclude
    return cfg
