import copy

import torch
from torch import nn

__all__ = ["Metric", "MetricCollection"]
__version__ = "1.0.0"


class Metric(nn.Module):
    """Minimal torchmetrics compatibility layer for Lightning logging.

    This project logs plain tensors, so Lightning only needs the base Metric
    protocol for its internal result wrappers.
    """

    def __init__(self) -> None:
        super().__init__()
        self._defaults = {}
        self._computed = None
        self._forward_cache = None
        self.update_called = False
        self._update_called = False

    def add_state(self, name, default, dist_reduce_fx=None, persistent=True) -> None:
        value = default.clone() if torch.is_tensor(default) else copy.deepcopy(default)
        default_copy = default.clone() if torch.is_tensor(default) else copy.deepcopy(default)
        object.__setattr__(self, name, value)
        self._defaults[name] = default_copy

    def reset(self) -> None:
        for name, default in self._defaults.items():
            value = default.clone() if torch.is_tensor(default) else copy.deepcopy(default)
            object.__setattr__(self, name, value)
        self._computed = None
        self._forward_cache = None
        self.update_called = False
        self._update_called = False

    def forward(self, *args, **kwargs):
        self.update_called = True
        self._update_called = True
        return self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class MetricCollection(dict):
    _enable_compute_groups = False

    def items(self, *args, **kwargs):
        return super().items()

