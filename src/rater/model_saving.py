import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch

from . import configs as cfg
from .utils import get_config_as_param


class AutoSave:
    def __init__(self, top_k=1, metric="f1", mode="min", root=None, name="ckpt"):
        self.top_k = top_k
        self.logs = []
        self.metric = metric
        self.mode = mode
        self.root = Path(root or cfg.MODEL_ROOT)
        assert self.root.exists()
        self.name = name

        self.top_models = []
        self.top_metrics = []

        self.slug_regex = r"[^\w_\-\.]"
        self.oof_suffix = "oof"

        self._log_path = None

    def slugify(self, s):
        return re.sub(self.slug_regex, "", s)

    def log(self, model, metrics, oof=None):
        metric = metrics[self.metric]
        rank = self.rank(metric)

        self.top_metrics.insert(rank + 1, metric)
        if len(self.top_metrics) > self.top_k:
            self.top_metrics.pop(0)

        self.logs.append(metrics)
        self.save(model, metric, rank, metrics["epoch"], oof=oof)

    def oof_path_from_model_path(self, model_path):
        oof_name = model_path.parent / "{}_{}.pkl".format(
            model_path.stem, self.oof_suffix
        )
        return oof_name

    def save(self, model, metric, rank, epoch, oof=None):
        t = time.strftime("%Y%m%d%H%M%S")
        name = "{}_epoch_{:02d}_{}_{:.04f}_{}".format(
            self.name, epoch, self.metric, metric, t
        )
        name = self.slugify(name) + ".pth"
        path = self.root.joinpath(name)

        old_model = None
        self.top_models.insert(rank + 1, name)
        if len(self.top_models) > self.top_k:
            old_model = self.root.joinpath(self.top_models[0])
            self.top_models.pop(0)

        torch.save(model.state_dict(), path.as_posix())

        if oof is not None:
            with self.oof_path_from_model_path(path).open(mode="wb") as f:
                pickle.dump(oof, f)

        if old_model is not None:
            old_model.unlink()
            old_oof = self.oof_path_from_model_path(old_model)
            if old_oof.exists():
                old_oof.unlink()

        self.to_json()

    def rank(self, val):
        r = -1
        for top_val in self.top_metrics:
            if val <= top_val:
                return r
            r += 1

        return r

    @property
    def log_path(self):
        if self._log_path is None:
            t = time.strftime("%Y%m%d%H%M%S")
            name = "{}_{}_logs".format(self.name, t)
            name = self.slugify(name) + ".json"
            self._log_path = self.root.joinpath(name)

        return self._log_path

    def to_json(self):

        with self.log_path.open(mode="w") as f:
            data = {
                "log": self.logs,
                "params": get_config_as_param(cfg),
            }
            json.dump(data, f, indent=2, default=default_json_converter)


def default_json_converter(x):
    converters = {
        np.ndarray: lambda x: x.tolist(),
        torch.Tensor: lambda x: x.detach().cpu().numpy().tolist(),
    }

    for dtype, converter in converters.items():
        if isinstance(x, dtype):
            return converter(x)

    return str(x)
