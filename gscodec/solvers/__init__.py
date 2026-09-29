# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE.audiocraft file in the root directory of this source tree.
"""
Solvers. A Solver is a training recipe, combining the dataloaders, models,
optimizer, losses etc into a single convenient object.
"""

# flake8: noqa
from .builders import get_solver
from .base import StandardSolver
from .compression import CompressionSolver
