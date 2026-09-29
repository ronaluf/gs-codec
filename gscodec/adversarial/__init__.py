# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE.audiocraft file in the root directory of this source tree.
"""Adversarial losses and discriminator architectures."""

# flake8: noqa
from .discriminators import (
    MultiScaleSTFTDiscriminator
)
from .losses import (
    AdversarialLoss,
    AdvLossType,
    get_adv_criterion,
    get_fake_criterion,
    get_real_criterion,
    FeatLossType,
    FeatureMatchingLoss
)
