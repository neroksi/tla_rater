import json
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from . import configs as cfg
from .utils import get_config_as_param


def default_json_converter(x):
    """
    Convert an object to a JSON-serializable format.

    Parameters
    ----------
    x : object
        The object to be converted.

    Returns
    -------
    str or list
        A JSON-serializable representation of the input object.
        - If `x` is a numpy array, it is converted to a list.
        - If `x` is a torch tensor, it is converted to a list after detaching from GPU and converting to a numpy array.
        - If `x` is of any other type, it is converted to its string representation.
    """
    # Dictionary of type-specific conversion functions
    converters = {
        np.ndarray: lambda x: x.tolist(),
        torch.Tensor: lambda x: x.detach().cpu().numpy().tolist(),
    }

    # Check the type of `x` and apply the appropriate converter
    for dtype, converter in converters.items():
        if isinstance(x, dtype):
            return converter(x)

    # If `x` is not of any specified type, convert it to a string
    return str(x)


class AutoSave:
    def __init__(
        self,
        top_k: int = 1,
        metric: str = "f1",
        mode: str = "min",
        root: Optional[str] = None,
        name: str = "ckpt",
    ):
        """
        Initialize the AutoSave instance.

        Parameters
        ----------
        top_k : int, optional
            Number of top models to keep, by default 1
        metric : str, optional
            Metric to monitor, by default "f1"
        mode : str, optional
            Mode for comparing metrics ("min" or "max"), by default "min"
        root : Optional[str], optional
            Root directory to save models, by default None
        name : str, optional
            Name prefix for saved models, by default "ckpt"
        """
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

    def slugify(self, s: str) -> str:
        """
        Convert string to a slug-friendly format.

        Parameters
        ----------
        s : str
            Input string

        Returns
        -------
        str
            Slugified string
        """
        return re.sub(self.slug_regex, "", s)

    def log(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, float],
        oof: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log the model and its metrics.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be saved
        metrics : Dict[str, float]
            Dictionary of metrics with their values
        oof : Optional[Dict[str, Any]], optional
            Out-of-fold predictions, by default None
        """
        metric = metrics[self.metric]
        rank = self.rank(metric)

        self.top_metrics.insert(rank + 1, metric)
        if len(self.top_metrics) > self.top_k:
            self.top_metrics.pop(0)

        self.logs.append(metrics)
        self.save(model, metric, rank, metrics["epoch"], oof=oof)

    def oof_path_from_model_path(self, model_path: Path) -> Path:
        """
        Generate out-of-fold path from model path.

        Parameters
        ----------
        model_path : Path
            Path to the saved model

        Returns
        -------
        Path
            Path for out-of-fold predictions
        """
        oof_name = model_path.parent / "{}_{}.pkl".format(
            model_path.stem, self.oof_suffix
        )
        return oof_name

    def save(
        self,
        model: torch.nn.Module,
        metric: float,
        rank: int,
        epoch: int,
        oof: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save the model and associated data.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be saved
        metric : float
            Metric value for the current model
        rank : int
            Rank of the current model based on the metric
        epoch : int
            Epoch number
        oof : Optional[Dict[str, Any]], optional
            Out-of-fold predictions, by default None
        """
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

    def rank(self, val: float) -> int:
        """
        Determine the rank of the given metric value.

        Parameters
        ----------
        val : float
            Metric value to rank

        Returns
        -------
        int
            Rank of the metric value
        """
        r = -1
        for top_val in self.top_metrics:
            if val <= top_val:
                return r
            r += 1

        return r

    @property
    def log_path(self) -> Path:
        """
        Get the path to the log file.

        Returns
        -------
        Path
            Path to the log file
        """
        if self._log_path is None:
            t = time.strftime("%Y%m%d%H%M%S")
            name = "{}_{}_logs".format(self.name, t)
            name = self.slugify(name) + ".json"
            self._log_path = self.root.joinpath(name)

        return self._log_path

    def to_json(self) -> None:
        """
        Save logs and parameters to a JSON file.
        """
        with self.log_path.open(mode="w") as f:
            data = {
                "log": self.logs,
                "params": get_config_as_param(cfg),
            }
            json.dump(data, f, indent=2, default=default_json_converter)
