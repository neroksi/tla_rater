import json
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import rater.configs as cfg
from rater.dataset import (
    Dataset,
    get_append_bar_chars,
    read_train_df,
)
from rater.script_utils import (
    get_config_as_param,
    get_special_token_ids,
    mp_gen_data,
    read_and_set_config,
)
from rater.training import _train, load_tokenizer
from transformers import AutoConfig


def prepare_data():
    df = read_train_df()

    if "id" not in df:
        df["id"] = df["essay_id_comp"]
    df["discourse_type_id"] = df["discourse_type_id"].fillna(-100).astype(int)

    with open(cfg.FOLD_JSON_PATH) as f:
        fold_dict = json.load(f)
    df[cfg.FOLD_COL_NAME] = df["id"].map(fold_dict).fillna(-1).astype(int)

    empty_uuids = df.loc[df["discourse_effectiveness"].isnull(), "id"].unique()
    df = df.loc[
        (df["discourse_type"] != "Unannotated") & ~df["id"].isin(empty_uuids)
    ].reset_index(drop=True)

    print("df.shape:", df.shape)

    print("FOLDS:\n", df[cfg.FOLD_COL_NAME].value_counts())

    uuid_with_folds = dict(df[["id", cfg.FOLD_COL_NAME]].drop_duplicates().values)

    return uuid_with_folds, df


def train(config_yaml_path: Path = None):
    read_and_set_config(config_yaml_path, cfg=cfg)

    print(
        "Folders Exist:",
        cfg.MODEL_ROOT.exists(),
        cfg.TRAIN_ROOT.exists(),
        cfg.TRAIN_CSV_PATH.exists(),
    )

    print(
        "OPTIMIZER:",
        cfg.OPTIMIZER_LR,
        cfg.OPTIMIZER_WEIGHT_DECAY,
        cfg.SCHEDULER_ETA_MIN,
    )

    N_CPU = cpu_count()
    print(f"n_cpu: {N_CPU}")

    config = AutoConfig.from_pretrained(cfg.MODEL_NAME)
    # tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    _, tokenizer = load_tokenizer(cfg.MODEL_NAME, config=config)

    APPEND_BAR_CHARS = get_append_bar_chars(tokenizer=tokenizer)
    print("APPEND_BAR_CHARS:", APPEND_BAR_CHARS)

    special_token_ids = get_special_token_ids(tokenizer)

    print("special_token_ids:\n", special_token_ids)

    uuid_with_folds, df = prepare_data()

    uuids = list(uuid_with_folds.keys())

    if cfg.IS_DEBUG:
        print("DEBUG MODE ENABLED")
        uuids = uuids[: cfg.NUM_DEBUG_ROWS]
        df = df.loc[df["id"].isin(uuids)]

    print("num uuids:", len(uuids))

    data = mp_gen_data(
        uuids=uuids, df=df, tokenizer=tokenizer, append_bar=APPEND_BAR_CHARS
    )

    doc_lens = pd.Series([len(x[2]) for x in data.values()])

    print("doc_lens:", doc_lens.min(), doc_lens.max())
    print(
        "doc lens quantiles\n:",
        doc_lens.quantile(
            np.concatenate([np.arange(0, 0.20, 0.025), np.arange(0.80, 1.00, 0.025)])
        ),
    )

    ds = Dataset(
        uuids,
        data,
        pad_token_id=special_token_ids["pad_token_id"],
        mask_token_id=special_token_ids["mask_token_id"],
        special_token_ids=special_token_ids,
    )

    print(f"len(ds): {len(ds)}")

    word_ids, input_ids, masks, target, eff_target = ds[0]

    print(
        "shapes:",
        word_ids.shape,
        input_ids.shape,
        masks.shape,
        target.shape,
        eff_target.shape,
    )

    config_dict = get_config_as_param(cfg)
    print("config_dict:", config_dict)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="The RATER Challenge Model Training Module"
    )

    parser.add_argument(
        "--config_yaml_path",
        type=str,
        required=True,
        help="Path to  the config YAML  file.",
    )

    args = parser.parse_args()

    print("CMD args:", args)

    train(args.config_yaml_path)
