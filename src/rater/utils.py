import os
import random
import re
from pathlib import Path

import numpy as np
import torch

from . import configs


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def slugify(s):
    return re.sub(r"[^\w\-_]", "_", s)


def get_config_as_param(configs):
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
    )

    # config_dict = {key: getattr(configs, key) for key in configs.__dir__()}
    config_dict = {key: getattr(configs, key) for key in dir(configs)}

    config_dict = {
        key: val
        for key, val in config_dict.items()
        if isinstance(val, config_ok_types) and not key.startswith("__")
    }

    return config_dict


def copy_param_to_configs(param, cfg=None, copy_all=False):
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
            if isinstance(getattr(cfg, cfg_attr), Path):
                val = Path(val)
            setattr(cfg, cfg_attr, val)
