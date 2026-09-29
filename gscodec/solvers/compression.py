# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Ron Aluf, Alon Canfi, Eliya Nachmani.
# This source code is licensed under the MIT license found in the
# LICENSE.audiocraft and LICENSE files in the root directory of this source tree.

import logging
import multiprocessing
from pathlib import Path
import typing as tp

import flashy
import omegaconf
import torch
from torch import nn

from . import base, builders
from .. import models, quantization
from ..models.builders import get_compression_model
from ..utils import checkpoint
from ..utils.samples.manager import SampleManager
from ..utils.utils import get_pool_executor


logger = logging.getLogger(__name__)


class CompressionSolver(base.StandardSolver):
    """Solver for compression task.

    The compression task combines a set of perceptual and objective losses
    to train an EncodecModel (composed of an encoder-decoder and a quantizer)
    to perform high fidelity audio reconstruction.
    """
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.rng: torch.Generator  # set at each epoch
        self.adv_losses = builders.get_adversarial_losses(self.cfg)
        self.aux_losses = nn.ModuleDict()
        self.info_losses = nn.ModuleDict()
        assert not cfg.fsdp.use, "FSDP not supported by CompressionSolver."
        loss_weights = dict()
        for loss_name, weight in self.cfg.losses.items():
            if loss_name in ['adv', 'feat']:
                for adv_name, _ in self.adv_losses.items():
                    loss_weights[f'{loss_name}_{adv_name}'] = weight
            elif loss_name == 'penalty':
                self.penalty_weight = weight
            elif weight > 0:
                self.aux_losses[loss_name] = builders.get_loss(loss_name, self.cfg)
                loss_weights[loss_name] = weight
            else:
                self.info_losses[loss_name] = builders.get_loss(loss_name, self.cfg)
        if not hasattr(self, 'penalty_weight'):
            self.penalty_weight = 1.0
        self.balancer = builders.get_balancer(loss_weights, self.cfg.balancer)
        logger.info(f"Loss weights (balancer): {loss_weights}")
        logger.info(f"Penalty weight: {self.penalty_weight}")
        self.register_stateful('adv_losses')
        self._continue_best_source_keys = self._continue_best_source_keys + ['adv_losses']

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        # best model is the last for the compression model
        return None

    def build_model(self):
        """Instantiate model and optimizer."""
        # Model and optimizer
        self.model = get_compression_model(self.cfg).to(self.device)
        gs_cfg = getattr(self.cfg, 'gaussian_splat', None)
        use_predictor = gs_cfg is not None and gs_cfg.get('amortized_predictor', False)
        if use_predictor:
            self._setup_predictor(gs_cfg)
        if use_predictor and gs_cfg.get('amortized_pos_lr_mult', 1.0) != 1.0:
            param_groups = self.model.quantizer.get_amortized_param_groups(float(self.cfg.optim.lr))
            self.optimizer = builders.get_optimizer(param_groups, self.cfg.optim)
        else:
            self.optimizer = builders.get_optimizer(
                (p for p in self.model.parameters() if p.requires_grad), self.cfg.optim)
        total_updates = self.cfg.optim.epochs * self.cfg.optim.updates_per_epoch
        self.lr_scheduler = builders.get_lr_scheduler(self.optimizer, self.cfg.schedule, total_updates) \
            if self.cfg.get('schedule') else None
        self.register_stateful('model', 'optimizer', 'lr_scheduler')
        self.register_best_state('model')
        self.register_ema('model')

    def _setup_predictor(self, gs_cfg: omegaconf.DictConfig):
        """Build the GS Predictor, load a trained codec and freeze it."""
        T = int(round(float(self.cfg.dataset.segment_duration) * float(self.model.frame_rate)))
        predictor = self.model.quantizer.build_predictor(T, self.device)
        logger.info("GS Predictor: T=%d, %d parameters", T, sum(p.numel() for p in predictor.parameters()))
        ckpt_path = gs_cfg.get('amortized_pretrained_codec_path', None)
        if not ckpt_path:
            return
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'best_state' in state:
            state = state['best_state']
        state = state.get('model', state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        missing = [k for k in missing if '.predictor.' not in k]
        if missing or unexpected:
            raise RuntimeError(f"Codec checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        logger.info("Loaded codec from %s", ckpt_path)
        if gs_cfg.get('amortized_freeze_codec', True):
            for name, p in self.model.named_parameters():
                if '.predictor.' not in name:
                    p.requires_grad_(False)

    def build_dataloaders(self):
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_audio_datasets(self.cfg)

    def show(self):
        """Show the compression model and employed adversarial loss."""
        self.logger.info(f"Compression model with {self.model.quantizer.total_codebooks} codebooks:")
        self.log_model_summary(self.model)
        self.logger.info("Adversarial loss:")
        self.log_model_summary(self.adv_losses)
        self.logger.info("Auxiliary losses:")
        self.logger.info(self.aux_losses)
        self.logger.info("Info losses:")
        self.logger.info(self.info_losses)

    def run_step(self, idx: int, batch: torch.Tensor, metrics: dict):
        """Perform one training or valid step on a given batch."""
        x = batch.to(self.device)
        y = x.clone()

        qres = self.model(x)
        assert isinstance(qres, quantization.QuantizedResult)
        y_pred = qres.x
        # Log bandwidth in kb/s
        metrics['bandwidth'] = qres.bandwidth.mean()

        if self.is_training:
            d_losses: dict = {}
            if len(self.adv_losses) > 0 and torch.rand(1, generator=self.rng).item() <= 1 / self.cfg.adversarial.every:
                for adv_name, adversary in self.adv_losses.items():
                    disc_loss = adversary.train_adv(y_pred, y)
                    d_losses[f'd_{adv_name}'] = disc_loss
                metrics['d_loss'] = torch.sum(torch.stack(list(d_losses.values())))
            metrics.update(d_losses)

        balanced_losses: dict = {}
        other_losses: dict = {}

        if qres.penalty is not None and qres.penalty.requires_grad:
            other_losses['penalty'] = qres.penalty

        # adversarial losses
        for adv_name, adversary in self.adv_losses.items():
            adv_loss, feat_loss = adversary(y_pred, y)
            balanced_losses[f'adv_{adv_name}'] = adv_loss
            balanced_losses[f'feat_{adv_name}'] = feat_loss

        # auxiliary losses
        for loss_name, criterion in self.aux_losses.items():
            loss = criterion(y_pred, y)
            balanced_losses[loss_name] = loss

        # weighted losses
        metrics.update(balanced_losses)
        metrics.update(other_losses)
        metrics.update(qres.metrics)

        if self.is_training:
            if 'penalty' in other_losses:
                penalty_loss = self.penalty_weight * other_losses['penalty']
                penalty_loss.backward(retain_graph=True)

            # balancer losses backward, returns effective training loss
            # with effective weights at the current batch.
            if len(balanced_losses) > 0:
                metrics['g_loss'] = self.balancer.backward(balanced_losses, y_pred)
                # add metrics corresponding to weight ratios
                metrics.update(self.balancer.metrics)
            else:
                metrics['g_loss'] = torch.tensor(0.0, device=y_pred.device)
            ratio2 = sum(p.grad.data.norm(p=2).pow(2)
                         for p in self.model.parameters() if p.grad is not None)
            assert isinstance(ratio2, torch.Tensor)
            metrics['ratio2'] = ratio2.sqrt()

            # optim
            flashy.distrib.sync_model(self.model)
            if self.cfg.optim.max_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.optim.max_norm
                )
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # informative losses only
        info_losses: dict = {}
        with torch.no_grad():
            for loss_name, criterion in self.info_losses.items():
                loss = criterion(y_pred, y)
                info_losses[loss_name] = loss

        metrics.update(info_losses)

        # aggregated GAN losses: this is useful to report adv and feat across different adversarial loss setups
        adv_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith('adv')]
        if len(adv_losses) > 0:
            metrics['adv'] = torch.sum(torch.stack(adv_losses))
        feat_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith('feat')]
        if len(feat_losses) > 0:
            metrics['feat'] = torch.sum(torch.stack(feat_losses))

        return metrics

    def run_epoch(self):
        # reset random seed at the beginning of the epoch
        self.rng = torch.Generator()
        self.rng.manual_seed(1234 + self.epoch)
        # run epoch
        super().run_epoch()

    def evaluate(self):
        """Evaluate stage. Runs audio reconstruction evaluation."""
        torch.cuda.empty_cache()
        self.model.eval()
        evaluate_stage_name = str(self.current_stage)

        loader = self.dataloaders['evaluate']
        updates = len(loader)
        lp = self.log_progress(f'{evaluate_stage_name} inference', loader, total=updates, updates=self.log_updates)
        average = flashy.averager()

        pendings = []
        ctx = multiprocessing.get_context('spawn')
        with get_pool_executor(self.cfg.evaluate.num_workers, mp_context=ctx) as pool:
            for idx, batch in enumerate(lp):
                x = batch.to(self.device)
                with torch.no_grad():
                    qres = self.model(x)

                y_pred = qres.x.cpu()
                y = batch.cpu()  # should already be on CPU but just in case
                pendings.append(pool.submit(evaluate_audio_reconstruction, y_pred, y, self.cfg))

            metrics_lp = self.log_progress(f'{evaluate_stage_name} metrics', pendings, updates=self.log_updates)
            for pending in metrics_lp:
                metrics = pending.result()
                metrics = average(metrics)

        metrics = flashy.distrib.average_metrics(metrics, len(loader))
        return metrics

    def generate(self):
        """Generate stage."""
        self.model.eval()
        sample_manager = SampleManager(self.xp, map_reference_to_sample_id=True)
        generate_stage_name = str(self.current_stage)

        loader = self.dataloaders['generate']
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        for batch in lp:
            reference, _ = batch
            reference = reference.to(self.device)
            with torch.no_grad():
                qres = self.model(reference)
            assert isinstance(qres, quantization.QuantizedResult)

            reference = reference.cpu()
            estimate = qres.x.cpu()
            sample_manager.add_samples(estimate, self.epoch, ground_truth_wavs=reference)

        flashy.distrib.barrier()

    @staticmethod
    def model_from_checkpoint(checkpoint_path: tp.Union[Path, str],
                              device: tp.Union[torch.device, str] = 'cpu') -> models.CompressionModel:
        """Instantiate a CompressionModel from a given checkpoint path or dora sig.
        This method is a convenient endpoint to load a CompressionModel to use in other solvers.

        Args:
            checkpoint_path (Path or str): Path to checkpoint or dora sig from where the checkpoint is resolved.
            device (torch.device or str): Device on which the model is loaded.
        """
        checkpoint_path = str(checkpoint_path)
        logger.info(f"Loading compression model from checkpoint: {checkpoint_path}")
        _checkpoint_path = checkpoint.resolve_checkpoint_path(checkpoint_path, use_fsdp=False)
        assert _checkpoint_path is not None, f"Could not resolve compression model checkpoint path: {checkpoint_path}"
        state = checkpoint.load_checkpoint(_checkpoint_path)
        assert state is not None and 'xp.cfg' in state, f"Could not load compression model from ckpt: {checkpoint_path}"
        cfg = state['xp.cfg']
        cfg.device = device
        compression_model = get_compression_model(cfg).to(device)
        assert compression_model.sample_rate == cfg.sample_rate, "Compression model sample rate should match"

        assert 'best_state' in state and state['best_state'] != {}
        if isinstance(compression_model.quantizer, quantization.GaussianSplatQuantizer) \
                and compression_model.quantizer.amortized_predictor_enabled:
            T = int(round(float(cfg.dataset.segment_duration) * float(compression_model.frame_rate)))
            compression_model.quantizer.build_predictor(T, device)
        compression_model.load_state_dict(state['best_state']['model'])
        compression_model.eval()
        logger.info("Compression model loaded!")
        return compression_model


def evaluate_audio_reconstruction(y_pred: torch.Tensor, y: torch.Tensor, cfg: omegaconf.DictConfig) -> dict:
    """Audio reconstruction evaluation method that can be conveniently pickled."""
    metrics = {}
    sisnr = builders.get_loss('sisnr', cfg)
    metrics['sisnr'] = sisnr(y_pred, y)

    if cfg.evaluate.metrics.get('pesq', False):
        try:
            from ..metrics.pesq import PesqMetric
            pesq_metric = PesqMetric(sample_rate=cfg.sample_rate)
            pesq_metric.update(y_pred, y)
            metrics['pesq'] = pesq_metric.compute().item()
        except Exception as e:
            logger.warning(f"PESQ computation failed: {e}")

    if cfg.evaluate.metrics.get('stoi', False):
        try:
            from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
            stoi = ShortTimeObjectiveIntelligibility(fs=cfg.sample_rate)
            metrics['stoi'] = stoi(y_pred, y).item()
        except Exception as e:
            logger.warning(f"STOI computation failed: {e}")

    if cfg.evaluate.metrics.get('utmos', False):
        try:
            import torchaudio
            from ..metrics.utmos import UTMOSScore

            if not hasattr(evaluate_audio_reconstruction, '_utmos_scorer'):
                evaluate_audio_reconstruction._utmos_scorer = UTMOSScore(device='cpu')
            recon_tensor = y_pred[0, 0].unsqueeze(0)
            if cfg.sample_rate != 16000:
                recon_16k = torchaudio.functional.resample(recon_tensor, cfg.sample_rate, 16000)
            else:
                recon_16k = recon_tensor
            utmos_score = evaluate_audio_reconstruction._utmos_scorer.score(recon_16k)
            metrics['utmos'] = utmos_score.item()
        except Exception as e:
            logger.warning(f"UTMOS computation failed: {e}")

    return metrics
