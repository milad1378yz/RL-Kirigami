"""Training package for Flow Matching and RL on the kirigami x-matrix task."""

import copy
import sys
import types

import torch
from torch import nn


def ensure_lightning_compat() -> None:
    """Install a minimal torchmetrics shim for Lightning when the env stack is broken.

    In the current Conda env, importing pytorch_lightning can fail because
    torchmetrics pulls in transformers/accelerate transitively. This project
    only logs plain tensors, so Lightning only needs the base Metric protocol.
    """

    if "torchmetrics" in sys.modules:
        return

    shim = types.ModuleType("torchmetrics")
    shim.__all__ = ["Metric", "MetricCollection"]
    shim.__version__ = "1.0.0"

    class Metric(nn.Module):
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

    shim.Metric = Metric
    shim.MetricCollection = MetricCollection
    sys.modules["torchmetrics"] = shim
