import math
import os
import re
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List
from warnings import filterwarnings

import joblib
import numpy as np
import torch
import yaml

from . import configs as cfg
from .dataset import (
    gen_data_from_id,
    read_from_id,
)
from .utils import copy_param_to_configs

filterwarnings("ignore", category=FutureWarning)


def get_fold(path):
    return int(re.search(r"fold.?(\d+)", str(path)).group(1))


def print_duration(d: float, desc=None):
    d = math.ceil(d)
    desc = desc or "DURATION"
    print(f"{desc}: {d//60:02d} mins {d%60:02d} s")


def read_and_set_config(yaml_config_path=None, cfg=None):
    if yaml_config_path is None:
        return

    if cfg is None:
        from . import configs as cfg

    with open(yaml_config_path, encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    copy_param_to_configs(conf, cfg=cfg, copy_all=True)

    cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("device:", cfg.DEVICE)

    return conf


def get_config_as_param(cfg):
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

    config_dict = {key: getattr(cfg, key) for key in cfg.__dir__()}

    config_dict = {
        key: val
        for key, val in config_dict.items()
        if isinstance(val, config_ok_types) and not key.startswith("__")
    }

    return config_dict


def get_special_token_ids(tokenizer):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.unk_token_id
    cls_token_id = tokenizer.cls_token_id

    assert pad_token_id >= 0
    assert mask_token_id >= 0
    # assert cls_token_id >= 0

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    special_token_ids = dict(
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        cls_token_id=cls_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )

    return special_token_ids


def take_sample(df, n_uuids=200):
    uuids = df["id"].unique()
    uuids = sorted(
        uuids,
        key=lambda uuid: -len(read_from_id(uuid, root=cfg.TRAIN_ROOT).split()),
    )[:n_uuids]
    df = df[df["id"].isin(uuids)]
    return df


def gen_data_from_ids(
    uuids, df, tokenizer, append_bar, root=None, texts: Dict[str, str] = None
):

    df_dict = {} if df is None else dict(list(df.groupby("id")))
    texts = {} if texts is None else texts

    data = {
        uuid: gen_data_from_id(
            uuid,
            df=df_dict.get(uuid),
            tokenizer=tokenizer,
            append_bar=append_bar,
            root=root,
            text=texts.get(uuid),
        )
        for uuid in uuids
    }
    return data


def mp_gen_data(
    uuids, df, tokenizer, append_bar, root=None, texts: Dict[str, str] = None
):
    T0 = time.time()
    N_CPU = cpu_count()

    uuids_list = [[] for _ in range(N_CPU)]
    for i, uuid in enumerate(uuids):
        uuids_list[i % N_CPU].append(uuid)

    mapper = joblib.delayed(gen_data_from_ids)
    tasks = [
        mapper(
            uuids_,
            df=None if df is None else df.loc[df["id"].isin(uuids_)],
            tokenizer=tokenizer,
            append_bar=append_bar,
            root=root,
            texts=None if texts is None else {uuid: texts[uuid] for uuid in uuids_},
        )
        for uuids_ in uuids_list
        if len(uuids_)
    ]
    res = joblib.Parallel(N_CPU)(tasks)

    data = {}
    for r in res:
        data.update(r)

    print_duration(time.time() - T0, "MP GEN DATA DURATION")

    return data
