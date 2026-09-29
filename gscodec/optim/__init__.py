# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE.audiocraft file in the root directory of this source tree.
"""Learning-rate schedulers and exponential moving average."""

# flake8: noqa
from .cosine_lr_scheduler import CosineLRScheduler
from .inverse_sqrt_lr_scheduler import InverseSquareRootLRScheduler
from .linear_warmup_lr_scheduler import LinearWarmupLRScheduler
from .polynomial_decay_lr_scheduler import PolynomialDecayLRScheduler
from .ema import ModuleDictEMA
