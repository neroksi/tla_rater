import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from . import configs


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def map_join(*args):
    return " ".join(map(str, args))


def slugify(s: str) -> str:
    """
    Converts a string into a slug by replacing non-alphanumeric characters with underscores.

    Parameters
    ----------
    s : str
        The input string to be slugified.

    Returns
    -------
    str
        The slugified version of the input string.
    """
    return re.sub(r"[^\w\-_]", "_", s)


def get_config_as_param(configs: Any) -> dict:
    """
    Extracts the configuration parameters from a given object.

    Parameters
    ----------
    configs : Any
        An object containing configuration parameters as attributes.

    Returns
    -------
    dict
        A dictionary of configuration parameters that are of acceptable types.
    """
    config_ok_types = (
        int,
        float,
        dict,
        str,
        tuple,
        list,
        np.ndarray,
        Path,
        torch.device,
        type(None),
    )

    # Extract attributes from the configs object
    config_dict = {key: getattr(configs, key) for key in dir(configs)}

    # Filter attributes to only include acceptable types and exclude private attributes
    config_dict = {
        key: val
        for key, val in config_dict.items()
        if isinstance(val, config_ok_types) and not key.startswith("__")
    }

    return config_dict


def copy_param_to_configs(
    param: Dict[str, Any], cfg: Optional[Any] = None, copy_all: bool = False
) -> None:
    """
    Copies parameters from a dictionary to a configuration object.

    Parameters
    ----------
    param : dict
        Dictionary of parameters to copy.
    cfg : Optional[Any], optional
        Configuration object to copy parameters to. If None, uses `configs`.
    copy_all : bool, optional
        If True, copies all parameters even if they do not exist in the configuration object.
    """
    cfg = configs if cfg is None else cfg
    for attr, val in param.items():

        cfg_attr = None
        if hasattr(cfg, attr):
            cfg_attr = attr
        elif hasattr(cfg, attr.upper()):
            cfg_attr = attr.upper()
        elif copy_all:
            cfg_attr = attr

        if cfg_attr is not None:
            # Ensure Path values are correctly converted
            if isinstance(getattr(cfg, cfg_attr), Path):
                val = Path(val)
            setattr(cfg, cfg_attr, val)
