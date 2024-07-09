import json
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import rater.configs as cfg
from rater.configs import logger
from rater.dataset import Dataset, read_train_df
from rater.script_utils import (
    get_config_as_param,
    get_special_token_ids,
    mp_gen_data,
    read_and_set_config,
)
from rater.training import load_tokenizer
from rater.training import train as _train
from rater.utils import map_join
from transformers import AutoConfig


def prepare_data() -> Tuple[Dict[str, int], pd.DataFrame]:
    """
    Prepare the training data by performing necessary preprocessing steps.

    Returns
    -------
    Tuple[Dict[str, int], pd.DataFrame]
        A tuple containing a dictionary mapping UUIDs to fold numbers, and the processed DataFrame.
    """
    # Read the training DataFrame
    df = read_train_df()

    # Ensure 'id' column is present
    if "id" not in df:
        df["id"] = df["essay_id_comp"]

    # Fill missing values in 'discourse_type_id' and convert to integer
    df["discourse_type_id"] = df["discourse_type_id"].fillna(-100).astype(int)

    # Load fold information from JSON file
    with open(cfg.FOLD_JSON_PATH) as f:
        fold_dict = json.load(f)

    # Map fold information to DataFrame
    df[cfg.FOLD_COL_NAME] = df["id"].map(fold_dict).fillna(-1).astype(int)

    # Identify and remove rows with missing discourse effectiveness
    empty_uuids = df.loc[df["discourse_effectiveness"].isnull(), "id"].unique()
    df = df.loc[
        (df["discourse_type"] != "Unannotated") & ~df["id"].isin(empty_uuids)
    ].reset_index(drop=True)

    # Print DataFrame shape and fold distribution
    logger.info(map_join("df.shape:", df.shape))
    logger.info(map_join("FOLDS:\n", df[cfg.FOLD_COL_NAME].value_counts()))

    # Create a dictionary of UUIDs with their corresponding folds
    uuid_with_folds = dict(df[["id", cfg.FOLD_COL_NAME]].drop_duplicates().values)

    return uuid_with_folds, df


def train(config_yaml_path: Path = None) -> None:
    """
    Train the model using the provided configuration.

    Parameters
    ----------
    config_yaml_path : Path, optional
        Path to the configuration YAML file, by default None
    """
    # Read and set configuration
    read_and_set_config(config_yaml_path, cfg=cfg)

    # Check if necessary folders exist
    logger.info(
        map_join(
            "Folders Exist:",
            cfg.MODEL_ROOT.exists(),
            cfg.TRAIN_ROOT.exists(),
            cfg.TRAIN_CSV_PATH.exists(),
        )
    )

    # Print optimizer configuration
    logger.info(
        map_join(
            "OPTIMIZER:",
            cfg.OPTIMIZER_LR,
            cfg.OPTIMIZER_WEIGHT_DECAY,
            cfg.SCHEDULER_ETA_MIN,
        )
    )

    # Get number of CPUs
    N_CPU = cpu_count()
    logger.info(map_join(f"n_cpu: {N_CPU}"))

    # Load model configuration and tokenizer
    config = AutoConfig.from_pretrained(cfg.MODEL_NAME)
    _, tokenizer = load_tokenizer(cfg.MODEL_NAME, config=config)
    special_token_ids = get_special_token_ids(tokenizer)

    logger.info(map_join("special_token_ids:\n", special_token_ids))

    # Prepare data
    uuid_with_folds, df = prepare_data()
    uuids = list(uuid_with_folds.keys())

    # Enable debug mode if configured
    if cfg.IS_DEBUG:
        logger.info("DEBUG MODE ENABLED")
        uuids = uuids[: cfg.NUM_DEBUG_ROWS]
        df = df.loc[df["id"].isin(uuids)]

    logger.info(map_join("num uuids:", len(uuids)))

    # Generate data for training
    data = mp_gen_data(uuids=uuids, df=df, tokenizer=tokenizer)

    # Calculate document lengths
    doc_lens = pd.Series([len(x[2]) for x in data.values()])
    logger.info(map_join("doc_lens:", doc_lens.min(), doc_lens.max()))
    logger.info(
        map_join(
            "doc lens quantiles\n:",
            doc_lens.quantile(
                np.concatenate(
                    [np.arange(0, 0.20, 0.025), np.arange(0.80, 1.00, 0.025)]
                )
            ),
        )
    )

    # Create dataset
    ds = Dataset(
        uuids,
        data,
        pad_token_id=special_token_ids["pad_token_id"],
        mask_token_id=special_token_ids["mask_token_id"],
        special_token_ids=special_token_ids,
    )

    logger.info(f"len(ds): {len(ds)}")

    # Print sample shapes
    word_ids, input_ids, masks, target, eff_target = ds[0]
    logger.info(
        map_join(
            "shapes:",
            word_ids.shape,
            input_ids.shape,
            masks.shape,
            target.shape,
            eff_target.shape,
        )
    )

    # Get configuration dictionary
    config_dict = get_config_as_param(cfg)
    logger.info(map_join("config_dict:", config_dict))

    # Train the model
    _train(
        save_each=cfg.SAVE_EACH,
        early_stop_epoch=cfg.EARLY_STOP_EPOCH,
        use_position_embeddings=False,
        use_stride_during_train=True,
        uuids={key: uuid_with_folds[key] for key in data},
        data=data,
        df=df,
        model_name=cfg.MODEL_NAME,
        pad_token_id=special_token_ids["pad_token_id"],
        mask_token_id=special_token_ids["mask_token_id"],
        tokenizer=tokenizer,
        model_config=config,
        epochs=cfg.EPOCHS,
        folds=cfg.FOLDS,
        suffix=f"_maxlen{cfg.MAXLEN}",
        special_token_ids=special_token_ids,
        checkpoint_paths=cfg.CHECKPOINT_PATHS,
    )


def main() -> None:
    """
    Main entry point for the RATER Challenge Model Training Module.

    Parses command line arguments and initiates the training process.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="The RATER Challenge Model Training Module"
    )

    parser.add_argument(
        "--config_yaml_path",
        type=str,
        required=True,
        help="Path to the config YAML file.",
    )

    args = parser.parse_args()

    logger.info(map_join("CMD args:", args))

    train(args.config_yaml_path)


if __name__ == "__main__":
    main()
