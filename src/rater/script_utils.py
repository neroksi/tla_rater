import math
import os
import re
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from warnings import filterwarnings

import joblib
import pandas as pd
import torch
import yaml

from . import configs as cfg
from .configs import logger
from .dataset import (
    gen_data_from_id,
    read_from_id,
)
from .utils import copy_param_to_configs, get_config_as_param, map_join

filterwarnings("ignore", category=FutureWarning)


def get_fold(path: Union[str, Path]) -> int:
    """
    Extract the fold number from the given path.

    Parameters
    ----------
    path : Union[str, Path]
        The file path from which to extract the fold number.

    Returns
    -------
    int
        The fold number extracted from the path.

    """
    return int(re.search(r"fold.?(\d+)", str(path)).group(1))


def print_duration(d: float, desc: str = None) -> None:
    """
    Prints the duration in a human-readable format (minutes and seconds).

    Parameters
    ----------
    d : float
        Duration in seconds.
    desc : str, optional
        Description to print before the duration. Defaults to "DURATION".
    """
    d = math.ceil(d)  # Round up the duration to the nearest second
    desc = desc or "DURATION"  # Use default description if none is provided
    logger.info(
        map_join(f"{desc}: {d//60:02d} mins {d%60:02d} s")
    )  # Print the duration in minutes and seconds


def read_and_set_config(
    yaml_config_path: Optional[str] = None, cfg: Optional[Any] = None
) -> Optional[dict]:
    """
    Reads a YAML configuration file and sets configuration parameters.

    Parameters
    ----------
    yaml_config_path : Optional[str], optional
        Path to the YAML configuration file, by default None
    cfg : Optional[ModuleType], optional
        Configuration module to set parameters, by default None

    Returns
    -------
    Optional[dict]
        The loaded configuration dictionary, or None if yaml_config_path is None.
    """
    if yaml_config_path is None:
        return None

    if cfg is None:
        from . import configs as cfg

    with open(yaml_config_path, encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    copy_param_to_configs(conf, cfg=cfg, copy_all=True)

    cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.info(("device:", cfg.DEVICE))

    return conf


def get_special_token_ids(tokenizer) -> dict:
    """
    Get the special token IDs from a tokenizer.

    Parameters
    ----------
    tokenizer : Any
        A tokenizer object with attributes for special tokens (pad, mask, cls, bos, eos).

    Returns
    -------
    dict
        A dictionary containing special token IDs.
    """
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    mask_token_id = tokenizer.mask_token_id
    if (mask_token_id is None) or (mask_token_id < 0):
        mask_token_id = tokenizer.unk_token_id
    cls_token_id = tokenizer.cls_token_id

    assert pad_token_id >= 0, "pad_token_id must be non-negative"
    assert mask_token_id >= 0, "mask_token_id must be non-negative"
    # cls_token_id can be None, so no assertion here

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


def take_sample(df: pd.DataFrame, n_uuids: int = 200) -> pd.DataFrame:
    """
    Takes a sample of the dataframe based on the length of text associated with unique IDs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column "id" with unique identifiers.
    n_uuids : int, optional
        Number of unique IDs to sample, by default 200.

    Returns
    -------
    pd.DataFrame
        A sampled DataFrame containing rows with the selected unique IDs.
    """
    uuids = df["id"].unique()
    uuids = sorted(
        uuids,
        key=lambda uuid: -len(read_from_id(uuid, root=cfg.TRAIN_ROOT).split()),
    )[:n_uuids]
    df = df[df["id"].isin(uuids)]
    return df


def gen_data_from_ids(
    uuids: List[str],
    df: pd.DataFrame,
    tokenizer: object,
    root: str = None,
    texts: Dict[str, str] = None,
    cfg_params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Generates data from a list of UUIDs by processing each UUID individually.

    Parameters
    ----------
    uuids : List[str]
        List of UUIDs to process.
    df : pd.DataFrame
        DataFrame containing data, grouped by 'id'.
    tokenizer : object
        Tokenizer object used for text processing.
    root : str, optional
        Root directory for data (default is None).
    texts : Dict[str, str], optional
        Dictionary of texts indexed by UUIDs (default is None).

    Returns
    -------
    Dict[str, Any]
        Dictionary with UUIDs as keys and processed data as values.
    """

    copy_param_to_configs(cfg_params, cfg=cfg)

    # Group dataframe by 'id' if df is provided, otherwise use an empty dictionary
    df_dict = {} if df is None else dict(list(df.groupby("id")))

    # Initialize texts as an empty dictionary if not provided
    texts = {} if texts is None else texts

    # Generate data for each UUID
    data = {
        uuid: gen_data_from_id(
            uuid,
            df=df_dict.get(uuid),
            tokenizer=tokenizer,
            root=root,
            text=texts.get(uuid),
        )
        for uuid in uuids
    }

    return data


def mp_gen_data(
    uuids: List[str],
    df: pd.DataFrame,
    tokenizer,
    root: str = None,
    texts: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Generate data in parallel using multiple CPU cores.

    Parameters
    ----------
    uuids : List[str]
        List of unique identifiers.
    df : pd.DataFrame
        DataFrame containing the data.
    tokenizer : object
        Tokenizer to be used for data processing.
    root : str, optional
        Root directory for data (default is None).
    texts : Dict[str, str], optional
        Dictionary containing text data (default is None).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the generated data.
    """
    T0 = time.time()
    N_CPU = cpu_count()  # Number of CPUs available
    cfg_params = get_config_as_param(cfg)

    # Split UUIDs into sublists for each CPU
    uuids_list = [[] for _ in range(N_CPU)]
    for i, uuid in enumerate(uuids):
        uuids_list[i % N_CPU].append(uuid)

    mapper = joblib.delayed(gen_data_from_ids)

    # Create tasks for parallel processing
    tasks = [
        mapper(
            uuids_,
            df=None if df is None else df.loc[df["id"].isin(uuids_)],
            tokenizer=tokenizer,
            root=root,
            texts=None if texts is None else {uuid: texts[uuid] for uuid in uuids_},
            cfg_params=cfg_params,
        )
        for uuids_ in uuids_list
        if len(uuids_)
    ]

    # Execute tasks in parallel
    res = joblib.Parallel(N_CPU)(tasks)

    data = {}
    for r in res:
        data.update(r)  # Combine results from all CPUs

    print_duration(time.time() - T0, "MP GEN DATA DURATION")

    return data
