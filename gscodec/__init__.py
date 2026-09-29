# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""GS-Codec: neural audio compression with Gaussian splatting."""

__version__ = "1.0.0"

from .codec import GSCode, GSCodec, build_model  # noqa: E402

__all__ = ["GSCodec", "GSCode", "build_model", "__version__"]
