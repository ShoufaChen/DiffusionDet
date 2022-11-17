# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import math
import itertools
import logging
from typing import Dict, Any
from contextlib import contextmanager

import torch
from detectron2.engine.train_loop import HookBase
from detectron2.checkpoint import DetectionCheckpointer


logger = logging.getLogger(__name__)


class EMADetectionCheckpointer(DetectionCheckpointer):
    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists
                and load the model together with all the checkpointables. Otherwise
                only load the model without loading any checkpointables.

        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            # workaround `self.load`
            return self.load(path, checkpointables=None)  # modify


class EMAState(object):
    def __init__(self):
        self.state = {}

    @classmethod
    def FromModel(cls, model: torch.nn.Module, device: str = ""):
        ret = cls()
        ret.save_from(model, device)
        return ret

    def save_from(self, model: torch.nn.Module, device: str = ""):
        """Save model state from `model` to this object"""
        for name, val in self.get_model_state_iterator(model):
            val = val.detach().clone()
            self.state[name] = val.to(device) if device else val

    def apply_to(self, model: torch.nn.Module):
        """Apply state to `model` from this object"""
        with torch.no_grad():
            for name, val in self.get_model_state_iterator(model):
                assert (
                    name in self.state
                ), f"Name {name} not existed, available names {self.state.keys()}"
                val.copy_(self.state[name])

    @contextmanager
    def apply_and_restore(self, model):
        old_state = EMAState.FromModel(model, self.device)
        self.apply_to(model)
        yield old_state
        old_state.apply_to(model)

    def get_ema_model(self, model):
        ret = copy.deepcopy(model)
        self.apply_to(ret)
        return ret

    @property
    def device(self):
        if not self.has_inited():
            return None
        return next(iter(self.state.values())).device

    def to(self, device):
        for name in self.state:
            self.state[name] = self.state[name].to(device)
        return self

    def has_inited(self):
        return self.state

    def clear(self):
        self.state.clear()
        return self

    def get_model_state_iterator(self, model):
        param_iter = model.named_parameters()
        buffer_iter = model.named_buffers()
        return itertools.chain(param_iter, buffer_iter)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict, strict: bool = True):
        self.clear()
        for x, y in state_dict.items():
            self.state[x] = y
        return torch.nn.modules.module._IncompatibleKeys(
            missing_keys=[], unexpected_keys=[]
        )

    def __repr__(self):
        ret = f"EMAState(state=[{','.join(self.state.keys())}])"
        return ret


class EMAUpdater(object):
    """Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and
    buffers). This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    Note:  It's very important to set EMA for ALL network parameters (instead of
    parameters that require gradient), including batch-norm moving average mean
    and variance.  This leads to significant improvement in accuracy.
    For example, for EfficientNetB3, with default setting (no mixup, lr exponential
    decay) without bn_sync, the EMA accuracy with EMA on params that requires
    gradient is 79.87%, while the corresponding accuracy with EMA on all params
    is 80.61%.

    Also, bn sync should be switched on for EMA.
    """

    def __init__(self, state: EMAState, decay: float = 0.999, device: str = "", yolox: bool = False):
        self.decay = decay
        self.device = device

        self.state = state
        self.updates = 0
        self.yolox = yolox
        if yolox:
            decay = 0.9998
            self.decay = lambda x: decay * (1 - math.exp(-x / 2000))

    def init_state(self, model):
        self.state.clear()
        self.state.save_from(model, self.device)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates) if self.yolox else self.decay
            for name, val in self.state.get_model_state_iterator(model):
                ema_val = self.state.state[name]
                if self.device:
                    val = val.to(self.device)
                ema_val.copy_(ema_val * d + val * (1.0 - d))


def add_model_ema_configs(_C):
    _C.MODEL_EMA = type(_C)()
    _C.MODEL_EMA.ENABLED = False
    _C.MODEL_EMA.DECAY = 0.999
    # use the same as MODEL.DEVICE when empty
    _C.MODEL_EMA.DEVICE = ""
    # When True, loading the ema weight to the model when eval_only=True in build_model()
    _C.MODEL_EMA.USE_EMA_WEIGHTS_FOR_EVAL_ONLY = False
    # when True, use YOLOX EMA: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/ema.py#L22
    _C.MODEL_EMA.YOLOX = False


def _remove_ddp(model):
    from torch.nn.parallel import DistributedDataParallel

    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def may_build_model_ema(cfg, model):
    if not cfg.MODEL_EMA.ENABLED:
        return
    model = _remove_ddp(model)
    assert not hasattr(
        model, "ema_state"
    ), "Name `ema_state` is reserved for model ema."
    model.ema_state = EMAState()
    logger.info("Using Model EMA.")


def may_get_ema_checkpointer(cfg, model):
    if not cfg.MODEL_EMA.ENABLED:
        return {}
    model = _remove_ddp(model)
    return {"ema_state": model.ema_state}


def get_model_ema_state(model):
    """Return the ema state stored in `model`"""
    model = _remove_ddp(model)
    assert hasattr(model, "ema_state")
    ema = model.ema_state
    return ema


def apply_model_ema(model, state=None, save_current=False):
    """Apply ema stored in `model` to model and returns a function to restore
    the weights are applied
    """
    model = _remove_ddp(model)

    if state is None:
        state = get_model_ema_state(model)

    if save_current:
        # save current model state
        old_state = EMAState.FromModel(model, state.device)
    state.apply_to(model)

    if save_current:
        return old_state
    return None


@contextmanager
def apply_model_ema_and_restore(model, state=None):
    """Apply ema stored in `model` to model and returns a function to restore
    the weights are applied
    """
    model = _remove_ddp(model)

    if state is None:
        state = get_model_ema_state(model)

    old_state = EMAState.FromModel(model, state.device)
    state.apply_to(model)
    yield old_state
    old_state.apply_to(model)


class EMAHook(HookBase):
    def __init__(self, cfg, model):
        model = _remove_ddp(model)
        assert cfg.MODEL_EMA.ENABLED
        assert hasattr(
            model, "ema_state"
        ), "Call `may_build_model_ema` first to initilaize the model ema"
        self.model = model
        self.ema = self.model.ema_state
        self.device = cfg.MODEL_EMA.DEVICE or cfg.MODEL.DEVICE
        self.ema_updater = EMAUpdater(
            self.model.ema_state, decay=cfg.MODEL_EMA.DECAY, device=self.device, yolox=cfg.MODEL_EMA.YOLOX
        )

    def before_train(self):
        if self.ema.has_inited():
            self.ema.to(self.device)
        else:
            self.ema_updater.init_state(self.model)

    def after_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        if not self.model.train:
            return
        self.ema_updater.update(self.model)
