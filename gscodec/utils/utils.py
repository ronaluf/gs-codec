# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE.audiocraft file in the root directory of this source tree.

from concurrent.futures import ProcessPoolExecutor
from functools import wraps
import hashlib
import logging
import typing as tp

import flashy
import flashy.distrib
import omegaconf
import torch
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


def model_hash(model: torch.nn.Module) -> str:
    """Return a model hash. This should allow us to track regressions in model init
    from the logs of past experiments.
    """
    hasher = hashlib.sha1()
    for p in model.parameters():
        hasher.update(p.data.cpu().numpy().tobytes())
    return hasher.hexdigest()


def dict_from_config(cfg: omegaconf.DictConfig) -> dict:
    """Convenience function to map an omegaconf configuration to a dictionary.

    Args:
        cfg (omegaconf.DictConfig): Original configuration to map to dict.
    Returns:
        dict: Config as dictionary object.
    """
    dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(dct, dict)
    return dct


def random_subset(dataset, max_samples: int, seed: int = 42) -> torch.utils.data.Subset:
    if max_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator)
    return torch.utils.data.Subset(dataset, perm[:max_samples].tolist())


def get_loader(dataset, num_samples: tp.Optional[int], batch_size: int,
               num_workers: int, seed: int, **kwargs) -> torch.utils.data.DataLoader:
    """Convenience function to load dataset into a dataloader with optional subset sampling.

    Args:
        dataset: Dataset to load.
        num_samples (Optional[int]): Number of samples to limit subset size.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.
        seed (int): Random seed.
    """
    if num_samples is not None:
        dataset = random_subset(dataset, num_samples, seed)

    dataloader = flashy.distrib.loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )
    return dataloader


class DummyPoolExecutor:
    """Dummy pool executor to use when we actually have only 1 worker.
    (e.g. instead of ProcessPoolExecutor).
    """
    class DummyResult:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers, mp_context=None):
        pass

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return


def get_pool_executor(num_workers: int, mp_context=None):
    return ProcessPoolExecutor(num_workers, mp_context) if num_workers > 1 else DummyPoolExecutor(1)


def with_rank_rng(base_seed: int = 1234):
    """Decorator for a function so that the function will use a Random Number Generator
    whose state depend on the GPU rank. The original RNG state is restored upon returning.

    Args:
        base_seed (int): Random seed.
    """
    def _decorator(fun: tp.Callable):
        @wraps(fun)
        def _decorated(*args, **kwargs):
            state = torch.get_rng_state()
            seed = base_seed ^ flashy.distrib.rank()
            torch.manual_seed(seed)
            logger.debug('Rank dependent seed set to %d', seed)
            try:
                return fun(*args, **kwargs)
            finally:
                torch.set_rng_state(state)
                logger.debug('RNG state restored.')
        return _decorated
    return _decorator


def collate(tensors: tp.List[torch.Tensor], dim: int = 0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Get a list of tensors and collate them to a single tensor. according to the following logic:
    - `dim` specifies the time dimension which will be stacked and padded.
    - The output will contain 1 new dimension (dimension index 0) which will be the size of
    of the original list.

    Args:
        tensors (tp.List[torch.Tensor]): List of tensors to collate.
        dim (int): Dimension which will be stacked and padded.
    Returns:
        tp.Tuple[torch.Tensor, torch.Tensor]:
            torch.Tensor: Stacked and padded tensor. The output will contain 1 new dimension
                (dimension index 0) which will be the size of the original list.
            torch.Tensor: Tensor containing length of original tensor sizes (without padding).
    """
    tensors = [x.transpose(0, dim) for x in tensors]
    lens = torch.LongTensor([len(x) for x in tensors])
    padded_tensors = pad_sequence(tensors)
    padded_tensors = padded_tensors.transpose(0, 1)
    padded_tensors = padded_tensors.transpose(1, dim + 1)
    return padded_tensors, lens


def copy_state(state: tp.Any, device: tp.Union[torch.device, str] = 'cpu',
               dtype: tp.Optional[torch.dtype] = None) -> tp.Any:
    if isinstance(state, torch.Tensor):
        if dtype is None or not state.is_floating_point():
            dtype = state.dtype
        return state.detach().to(device=device, dtype=dtype, copy=True)
    elif isinstance(state, dict):
        return {k: copy_state(v, device, dtype) for k, v in state.items()}
    elif isinstance(state, list):
        return [copy_state(v, device, dtype) for v in state]


