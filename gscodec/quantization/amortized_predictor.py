# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""GS Predictor: regresses Gaussian-splatting parameters in a single forward pass."""

import math
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pe_1d(length: int, d_model: int) -> torch.Tensor:
    """Sinusoidal positional encoding of shape ``(length, d_model)``."""
    pe = torch.zeros(length, d_model)
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class _DilatedConvBlock(nn.Module):
    def __init__(self, d_model: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        res = h
        h = self.act(self.conv(h))
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        return h + res


class _DecoderLayer(nn.Module):
    """Transformer decoder layer: self-attention, cross-attention, FFN."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        sa_out, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.norm1(q + sa_out)
        ca_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        q = self.norm2(q + ca_out)
        return self.norm3(q + self.ffn(q))


class AmortizedGSPredictor(nn.Module):
    """Predicts ``weights (B, C, N)``, ``scales (B, 1, N)`` and ``pos (B, 1, N)``."""

    def __init__(self, in_dim: int, n_gaussians: int, T: int, d_model: int = 128, n_heads: int = 4,
                 n_blocks: int = 4, n_decoder_layers: int = 1, pos_zero_init: bool = True,
                 scale_bias_mult: float = 1.0, variable_ng: bool = False, mode_embedding: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.n_gaussians = n_gaussians
        self.T = T
        self.d_model = d_model
        self.variable_ng = variable_ng
        self.mode_embedding = mode_embedding

        self.input_proj = nn.Conv1d(in_dim, d_model, kernel_size=1)
        self.backbone = nn.ModuleList([_DilatedConvBlock(d_model, dilation=2 ** i) for i in range(n_blocks)])
        self.register_buffer('backbone_pe', _sinusoidal_pe_1d(T, d_model).T.unsqueeze(0))

        if variable_ng:
            self.shared_query = nn.Parameter(torch.randn(d_model) * (1.0 / math.sqrt(d_model)))
            self.query_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        else:
            self.queries = nn.Parameter(torch.randn(n_gaussians, d_model) * (1.0 / math.sqrt(d_model)))
        grid_positions = torch.linspace(0.0, float(T - 1), n_gaussians)
        self.register_buffer('query_pe', self._grid_pe(grid_positions))
        self.register_buffer('grid_positions', grid_positions)

        if mode_embedding:
            self.mode_emb = nn.Embedding(2, d_model)
            nn.init.normal_(self.mode_emb.weight, std=1.0 / math.sqrt(d_model))

        self.layers = nn.ModuleList([_DecoderLayer(d_model, n_heads) for _ in range(n_decoder_layers)])
        self.head_weights = nn.Linear(d_model, in_dim)
        self.head_log_scale = nn.Linear(d_model, 1)
        self.head_pos = nn.Linear(d_model, 1)

        target_scale = (T / n_gaussians) * 3.0 * scale_bias_mult
        scale_bias = math.log(math.expm1(target_scale)) if target_scale < 20 else target_scale
        self.head_log_scale.bias.data.fill_(scale_bias)
        nn.init.zeros_(self.head_weights.bias)
        if pos_zero_init:
            nn.init.zeros_(self.head_pos.weight)
            nn.init.zeros_(self.head_pos.bias)

    def _grid_pe(self, grid_positions: torch.Tensor) -> torch.Tensor:
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32, device=grid_positions.device)
                        * (-math.log(10000.0) / self.d_model))
        q_pe = torch.zeros(grid_positions.shape[0], self.d_model, device=grid_positions.device)
        q_pe[:, 0::2] = torch.sin(grid_positions.unsqueeze(1) * div)
        q_pe[:, 1::2] = torch.cos(grid_positions.unsqueeze(1) * div)
        return q_pe

    def _make_queries(self, B: int, n_gaussians: tp.Optional[int], frozen_mode: tp.Optional[bool],
                      device: torch.device) -> torch.Tensor:
        if self.variable_ng:
            N = n_gaussians if n_gaussians is not None else self.n_gaussians
            if N == self.n_gaussians:
                q_pe = self.query_pe
            else:
                q_pe = self._grid_pe(torch.linspace(0.0, float(self.T - 1), N, device=device))
            q = self.shared_query.unsqueeze(0) + q_pe + self.query_mlp(q_pe)
        else:
            if n_gaussians is not None and n_gaussians != self.n_gaussians:
                raise ValueError(f"This predictor was trained for N_G={self.n_gaussians}, got {n_gaussians}.")
            q = self.queries + self.query_pe
        q = q.unsqueeze(0).expand(B, -1, -1)
        if self.mode_embedding and frozen_mode is not None:
            q = q + self.mode_emb.weight[int(frozen_mode)].view(1, 1, -1)
        return q

    def forward(self, x_norm: torch.Tensor, n_gaussians: tp.Optional[int] = None,
                frozen_mode: tp.Optional[bool] = None) -> tp.Dict[str, torch.Tensor]:
        B, D, T = x_norm.shape
        if D != self.in_dim or T != self.T:
            raise ValueError(f"Expected latent of shape (B, {self.in_dim}, {self.T}), got {tuple(x_norm.shape)}.")
        h = self.input_proj(x_norm)
        for block in self.backbone:
            h = block(h)
        h = h + self.backbone_pe
        q = self._make_queries(B, n_gaussians, frozen_mode, x_norm.device)
        kv = h.transpose(1, 2)
        for layer in self.layers:
            q = layer(q, kv)
        weights = self.head_weights(q).transpose(1, 2)
        scales = F.softplus(self.head_log_scale(q).transpose(1, 2))
        pos = torch.sigmoid(self.head_pos(q).transpose(1, 2)) * (self.T - 1)
        return {'weights': weights, 'scales': scales, 'pos': pos}


class AmortizedPredictorMixin:
    """GS Predictor methods of ``GaussianSplatQuantizer``."""

    dimension: int
    n_gaussians: int
    freeze_positions: bool
    predictor: tp.Optional[nn.Module]

    def _build_predictor(self, T: int, device: torch.device) -> nn.Module:
        net = AmortizedGSPredictor(
            in_dim=self.dimension,
            n_gaussians=self.n_gaussians,
            T=T,
            d_model=self.amortized_d_model,
            n_heads=self.amortized_n_heads,
            n_blocks=self.amortized_n_blocks,
            n_decoder_layers=self.amortized_n_decoder_layers,
            pos_zero_init=self.amortized_pos_zero_init,
            scale_bias_mult=self.amortized_scale_bias_mult,
            variable_ng=self.amortized_variable_ng,
            mode_embedding=self.amortized_mode_sampling,
        )
        return net.to(device)

    def build_predictor(self, T: int, device: tp.Union[str, torch.device] = "cpu") -> nn.Module:
        """Instantiate the predictor for latents of length ``T``."""
        if self.predictor is None:
            self.predictor = self._build_predictor(T, torch.device(device))
        return self.predictor

    def get_amortized_param_groups(self, base_lr: float) -> tp.List[tp.Dict]:
        """Optimizer groups applying ``amortized_pos_lr_mult`` to the center head."""
        if self.predictor is None or self.amortized_pos_lr_mult == 1.0:
            return [{'params': self.parameters()}]
        pos_ids = {id(p) for p in self.predictor.head_pos.parameters()}
        other = [p for p in self.predictor.parameters() if id(p) not in pos_ids]
        pos = [p for p in self.predictor.parameters() if id(p) in pos_ids]
        return [{'params': other, 'lr': base_lr},
                {'params': pos, 'lr': base_lr * self.amortized_pos_lr_mult}]

    def predict_params_amortized(self, x_norm: torch.Tensor, T: int, n_gaussians: tp.Optional[int] = None,
                                 frozen_mode: tp.Optional[bool] = None) -> tp.Dict[str, torch.Tensor]:
        """Predict ``weights``, ``scales`` and ``pos`` for a latent ``(B, C, T)``."""
        if self.predictor is None:
            self.predictor = self._build_predictor(T, x_norm.device)
        params = self.predictor(x_norm, n_gaussians=n_gaussians, frozen_mode=frozen_mode)
        if self.training and self.amortized_param_noise > 0.0:
            params['weights'] = params['weights'] + torch.randn_like(params['weights']) * self.amortized_param_noise
            params['scales'] = params['scales'] + torch.randn_like(params['scales']) * self.amortized_param_noise * 0.1
        return params

    def forward_amortized(self, x: torch.Tensor, x_norm: torch.Tensor, x_mean: torch.Tensor,
                          x_std: torch.Tensor, T: int, frame_rate: int):
        from .base import QuantizedResult

        B, D, _ = x_norm.shape
        target = x_norm.detach()

        n_override: tp.Optional[int] = None
        if (self.amortized_variable_ng and self.training
                and self.amortized_ng_min is not None and self.amortized_ng_max is not None):
            n_override = int(torch.randint(self.amortized_ng_min, self.amortized_ng_max + 1, (1,)).item())
        frozen = bool(self.freeze_positions)
        if self.amortized_mode_sampling and self.training:
            frozen = bool(torch.rand(1).item() < self.amortized_frozen_mode_prob)
        mode_hint = frozen if self.amortized_mode_sampling else None

        pred = self.predict_params_amortized(target, T, n_gaussians=n_override, frozen_mode=mode_hint)
        N = pred['pos'].shape[-1]
        if frozen:
            means = torch.linspace(0.0, float(T - 1), N, device=target.device, dtype=target.dtype
                                   ).view(1, 1, N).expand(B, 1, N).contiguous()
        else:
            means = pred['pos']
        params = {'means': means, 'scales': pred['scales'], 'weights': pred['weights'],
                  'bias': torch.zeros(B, D, device=target.device)}
        x_hat_norm = self._splat(params, T)
        latent_loss = F.mse_loss(x_hat_norm, target)

        x_hat = x_hat_norm * x_std + x_mean
        if self.amortized_freeze_codec:
            x_hat = x_hat.detach()
        bandwidth = torch.tensor((self.dimension * N + 2 * N) * 32 * frame_rate / 1000 / T, device=x.device)
        return QuantizedResult(x=x_hat, codes=x_hat.unsqueeze(1), bandwidth=bandwidth,
                               penalty=latent_loss, metrics={'amortized_mse': latent_loss.detach()})
