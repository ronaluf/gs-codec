# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE.audiocraft file in the root directory of this source tree.
"""Quantizers."""
# flake8: noqa
from .base import BaseQuantizer, DummyQuantizer, QuantizedResult
from .gaussian_splat import GaussianSplatQuantizer
