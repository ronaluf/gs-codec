# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Gaussian-splatting bottleneck: Z_hat[c, t] = sum_i w[i, c] * exp(-(t - mu_i)^2 / (2 sigma_i^2))."""

import typing as tp

import torch
import torch.nn.functional as F

from .amortized_predictor import AmortizedPredictorMixin
from .base import BaseQuantizer, QuantizedResult


GSParams = tp.Dict[str, torch.Tensor]


class GaussianSplatQuantizer(AmortizedPredictorMixin, BaseQuantizer):
    """Gaussian-splatting bottleneck.

    Args:
        dimension: Latent channels ``C``.
        n_gaussians: Number of primitives ``N_G`` per segment.
        n_iters: Inner Adam steps ``K``.
        lr: Inner Adam learning rate for amplitudes.
        lr_position_mult: Learning-rate multiplier for centers and scales.
        lr_schedule: ``"none"``, ``"cosine"``, ``"linear"`` or ``"exponential"``.
        lr_final_ratio: Final learning rate as a fraction of ``lr``.
        lr_warmup_iters: Warm-up steps.
        shared_positions: Share centers across channels.
        shared_scales: Share scales across channels.
        freeze_positions: Keep centers on the uniform grid.
        freeze_scales: Keep scales at their initial value.
        freeze_bias: Keep the per-channel bias at zero.
        normalize_input: Standardize each latent channel before fitting.
        warmup_mode: Bypass the bottleneck.
        param_clamp_*: Clamp ranges for scales and amplitudes during fitting.
        amortized_*: GS Predictor settings.
    """

    def __init__(
        self,
        dimension: int = 32,
        n_gaussians: int = 80,
        n_iters: int = 100,
        lr: float = 0.4,
        lr_position_mult: float = 1.0,
        lr_schedule: str = "none",
        lr_final_ratio: float = 0.1,
        lr_warmup_iters: int = 0,
        shared_positions: bool = True,
        shared_scales: tp.Optional[bool] = None,
        freeze_positions: bool = False,
        freeze_scales: bool = False,
        freeze_bias: bool = False,
        normalize_input: bool = False,
        warmup_mode: bool = False,
        param_clamp_scale_min: tp.Optional[float] = None,
        param_clamp_scale_max: tp.Optional[float] = None,
        param_clamp_weight_min: tp.Optional[float] = None,
        param_clamp_weight_max: tp.Optional[float] = None,
        amortized_predictor: bool = False,
        amortized_d_model: int = 128,
        amortized_n_heads: int = 4,
        amortized_n_blocks: int = 4,
        amortized_n_decoder_layers: int = 1,
        amortized_pos_zero_init: bool = True,
        amortized_scale_bias_mult: float = 1.0,
        amortized_variable_ng: bool = False,
        amortized_ng_min: tp.Optional[int] = None,
        amortized_ng_max: tp.Optional[int] = None,
        amortized_mode_sampling: bool = False,
        amortized_frozen_mode_prob: float = 0.5,
        amortized_param_noise: float = 0.0,
        amortized_pos_lr_mult: float = 1.0,
        amortized_pretrained_codec_path: tp.Optional[str] = None,
        amortized_freeze_codec: bool = True,
    ):
        super().__init__()
        if lr_schedule not in ("none", "cosine", "linear", "exponential"):
            raise ValueError(f"Unknown lr_schedule: {lr_schedule}")
        if not 0.0 <= amortized_frozen_mode_prob <= 1.0:
            raise ValueError("amortized_frozen_mode_prob must be in [0, 1]")
        self.dimension = dimension
        self.n_gaussians = n_gaussians
        self.n_iters = n_iters
        self.lr = lr
        self.lr_position_mult = lr_position_mult
        self.lr_schedule = lr_schedule
        self.lr_final_ratio = lr_final_ratio
        self.lr_warmup_iters = lr_warmup_iters
        self.shared_positions = shared_positions
        self.shared_scales = shared_positions if shared_scales is None else shared_scales
        self.freeze_positions = freeze_positions
        self.freeze_scales = freeze_scales
        self.freeze_bias = freeze_bias
        self.normalize_input = normalize_input
        self.warmup_mode = warmup_mode
        self.param_clamp_scale_min = param_clamp_scale_min
        self.param_clamp_scale_max = param_clamp_scale_max
        self.param_clamp_weight_min = param_clamp_weight_min
        self.param_clamp_weight_max = param_clamp_weight_max

        self.amortized_predictor_enabled = amortized_predictor
        self.amortized_d_model = amortized_d_model
        self.amortized_n_heads = amortized_n_heads
        self.amortized_n_blocks = amortized_n_blocks
        self.amortized_n_decoder_layers = amortized_n_decoder_layers
        self.amortized_pos_zero_init = amortized_pos_zero_init
        self.amortized_scale_bias_mult = amortized_scale_bias_mult
        self.amortized_variable_ng = amortized_variable_ng
        self.amortized_ng_min = amortized_ng_min
        self.amortized_ng_max = amortized_ng_max
        self.amortized_mode_sampling = amortized_mode_sampling
        self.amortized_frozen_mode_prob = float(amortized_frozen_mode_prob)
        self.amortized_param_noise = amortized_param_noise
        self.amortized_pos_lr_mult = amortized_pos_lr_mult
        self.amortized_pretrained_codec_path = amortized_pretrained_codec_path
        self.amortized_freeze_codec = amortized_freeze_codec
        self.predictor: tp.Optional[torch.nn.Module] = None

        torch.backends.cudnn.benchmark = True
        self._splat_compiled: tp.Optional[tp.Callable] = None
        self._compile_attempted = False

    def forward(self, x: torch.Tensor, frame_rate: int) -> QuantizedResult:
        B, D, T = x.shape
        x_norm, x_mean, x_std = self._normalize(x)

        if self.warmup_mode:
            return QuantizedResult(x=x, codes=x.unsqueeze(1),
                                   bandwidth=torch.tensor(0.0, device=x.device), penalty=None)

        if self.amortized_predictor_enabled:
            return self.forward_amortized(x, x_norm, x_mean, x_std, T, frame_rate)

        target = x_norm.detach()
        with torch.enable_grad():
            params = self._init_params(B, D, T, x.device, target=target)
            params = self._optimize(params, target, T)
            x_hat_norm = self._get_splat_fn(x.device)(params, T).clone()

        x_hat_opt = x_hat_norm * x_std + x_mean
        x_hat = x + (x_hat_opt - x).detach()
        commit_loss = F.mse_loss(x, x_hat_opt.detach())

        bandwidth = torch.tensor(
            self._count_params() * 32 * frame_rate / 1000 / T, device=x.device)
        return QuantizedResult(
            x=x_hat,
            codes=x_hat.unsqueeze(1),
            bandwidth=bandwidth,
            penalty=commit_loss,
            metrics={"commit_loss": commit_loss.detach()},
        )

    def _normalize(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.normalize_input:
            x_mean = x.mean(dim=-1, keepdim=True)
            x_std = x.std(dim=-1, keepdim=True) + 1e-6
            return (x - x_mean) / x_std, x_mean, x_std
        return x, torch.zeros(1, device=x.device), torch.ones(1, device=x.device)

    def _init_params(self, B: int, D: int, T: int, device: torch.device,
                     target: tp.Optional[torch.Tensor] = None) -> GSParams:
        """Uniform-grid centers, constant scales, amplitudes sampled from the target."""
        N = self.n_gaussians
        pos_D = 1 if self.shared_positions else D
        init_scale = T / N / 2
        means = torch.linspace(0, T - 1, N, device=device).view(1, 1, N).repeat(B, pos_D, 1)
        scales = torch.full((B, pos_D, N), init_scale, device=device)

        bias = torch.zeros(B, D, device=device)
        weights = torch.zeros(B, D, N, device=device)
        if target is not None:
            residual = target
            if not self.freeze_bias:
                bias = target.mean(dim=-1)
                residual = target - bias.unsqueeze(-1)
            idx = torch.linspace(0, T - 1, N, device=device).long()
            weights = residual[:, :, idx] * (1.0 / max(1.0, N / T * init_scale * 2.5))

        return {
            "means": means if self.freeze_positions else means.requires_grad_(True),
            "scales": scales.clone() if self.freeze_scales else scales.clone().requires_grad_(True),
            "weights": weights.clone().requires_grad_(True),
            "bias": bias.clone() if self.freeze_bias else bias.clone().requires_grad_(True),
        }

    def _build_optimizer(self, params: GSParams) -> torch.optim.Adam:
        amplitude_params = [params["weights"]]
        if params["bias"].requires_grad:
            amplitude_params.append(params["bias"])
        use_fused = params["means"].is_cuda
        position_params = []
        if not self.freeze_positions:
            position_params.append(params["means"])
        if not self.freeze_scales:
            position_params.append(params["scales"])
        groups = [{"params": amplitude_params, "lr": self.lr}]
        if position_params:
            groups.insert(0, {"params": position_params, "lr": self.lr * self.lr_position_mult})
        return torch.optim.Adam(groups, fused=use_fused, foreach=not use_fused)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer
                         ) -> tp.Optional[torch.optim.lr_scheduler.LRScheduler]:
        if self.lr_schedule == "none" or self.n_iters <= 1:
            return None
        decay_iters = max(1, self.n_iters - self.lr_warmup_iters)
        if self.lr_schedule == "exponential":
            base = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.lr_final_ratio ** (1.0 / decay_iters))
        elif self.lr_schedule == "cosine":
            base = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=decay_iters, eta_min=self.lr * self.lr_final_ratio)
        else:
            def linear_lr(step):
                if step >= decay_iters:
                    return self.lr_final_ratio
                return 1.0 - (1.0 - self.lr_final_ratio) * (step / decay_iters)
            base = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lr)
        if self.lr_warmup_iters <= 0:
            return base

        def warmup_lr(step):
            if step < self.lr_warmup_iters:
                return self.lr_final_ratio + (1.0 - self.lr_final_ratio) * step / self.lr_warmup_iters
            return 1.0
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, base], milestones=[self.lr_warmup_iters])

    def _optimize(self, params: GSParams, target: torch.Tensor, T: int) -> GSParams:
        """Fit the primitives to ``target`` with ``n_iters`` Adam steps."""
        optimizer = self._build_optimizer(params)
        scheduler = self._build_scheduler(optimizer)
        splat_fn = self._get_splat_fn(target.device)
        s_min = self.param_clamp_scale_min if self.param_clamp_scale_min is not None else 0.5
        s_max = self.param_clamp_scale_max if self.param_clamp_scale_max is not None else T / 4
        for _ in range(self.n_iters):
            optimizer.zero_grad()
            loss = F.mse_loss(splat_fn(params, T), target)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                if not self.freeze_positions:
                    params["means"].clamp_(min=0, max=T - 1)
                if not self.freeze_scales:
                    params["scales"].clamp_(min=s_min, max=s_max)
                if self.param_clamp_weight_min is not None or self.param_clamp_weight_max is not None:
                    params["weights"].clamp_(min=self.param_clamp_weight_min,
                                             max=self.param_clamp_weight_max)
        return {k: v.detach() for k, v in params.items()}

    def _get_splat_fn(self, device: torch.device) -> tp.Callable:
        if not self._compile_attempted and device.type == "cuda":
            self._compile_attempted = True
            self._splat_compiled = torch.compile(self._splat, mode="reduce-overhead")
        return self._splat_compiled if self._splat_compiled is not None else self._splat

    def _splat(self, params: GSParams, T: int) -> torch.Tensor:
        """Render primitives to a latent of shape ``(B, C, T)``."""
        B = params["means"].shape[0]
        D = params["weights"].shape[1]
        N = params["weights"].shape[2]
        t = torch.arange(T, device=params["means"].device, dtype=torch.float32).view(1, 1, 1, T)
        means = params["means"]
        scales = params["scales"]
        if self.shared_positions:
            means = means.expand(B, D, N)
        if self.shared_scales:
            scales = scales.expand(B, D, N)
        basis = torch.exp(-0.5 * ((t - means.unsqueeze(-1)) / scales.unsqueeze(-1)) ** 2)
        recon = (params["weights"].unsqueeze(-1) * basis).sum(dim=2)
        return recon + params["bias"].unsqueeze(-1)

    def render(self, params: GSParams, T: int) -> torch.Tensor:
        """Render parameters to a latent of shape ``(B, C, T)``."""
        return self._splat(params, T)

    def _count_params(self, N: tp.Optional[int] = None) -> int:
        N = self.n_gaussians if N is None else N
        D = self.dimension
        count = N * 2 if self.shared_positions else N * D * 2
        return count + N * D + D

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            q_res = self.forward(x.requires_grad_(False), frame_rate=1)
        return q_res.codes.detach()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.squeeze(1)

    @property
    def total_codebooks(self) -> int:
        return 1

    @property
    def num_codebooks(self) -> int:
        return 1

    @property
    def bins(self) -> int:
        return -1

    def set_num_codebooks(self, n: int) -> None:
        pass
