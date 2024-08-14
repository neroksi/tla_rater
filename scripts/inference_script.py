import gc
import os
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from warnings import filterwarnings

import joblib
import numpy as np
import pandas as pd
import rater.inference as inference
from rater import configs
from rater.configs import logger
from rater.post_processing import get_seg_from_ner
from rater.script_utils import print_duration, read_and_set_config
from rater.utils import map_join
from rater.wbf import fusion_boxes_for_subs

os.environ["TOKENIZERS_PARALLELISM"] = "false"
N_CPU = cpu_count()

for cat in [FutureWarning, UserWarning]:
    filterwarnings("ignore", category=cat)


class InfConfig:
    """
    A namespace to group main inference parameters. Any value here will be overrided by
    the values form yaml_config_file (if any).
    """

    IS_DEBUG = True  # Set to True for faster testing
    NUM_DEBUG_ROWS = 80  # How many rows to read when in DEBUG mode ?

    DEVICE = "cpu"  # Don't care too much, this will automtically be replaced by cuda device if available
    # Please see rater.script_utils:read_and_set_config() to change this behaviour

    IOU_THR = 0.3333  # Used for essay-spans ensembling ==> see ensemble-boxes package documentation for more background
    SKIP_BOX_THR = 0.01  # Same role as IOU_THR

    MODEL_PARAMS = []  # This should contain all the active model weights for inference

    TEST_CSV_PATH: str = "data/test.csv"
    SUB_CSV_SAVE_PATH: str = "data/submission.csv"
    LOGS_TXT_PATH = configs.LOGS_TXT_PATH

    MIN_LEN_THRESH_FOR_WBF = {  # minimum length for each detected class-span
        "Lead": 3,
        "Position": 4,
        "Evidence": 4,
        "Claim": 2,
        "Concluding Statement": 9,
        "Counterclaim": 5,
        "Rebuttal": 2,
    }

    PROBA_THRESH_FOR_WBF = {  # minimum probas for each detected class-span
        "Lead": 0.27,
        "Position": 0.28,
        "Evidence": 0.39,
        "Claim": 0.30,
        "Concluding Statement": 0.36,
        "Counterclaim": 0.21,
        "Rebuttal": 0.20,
    }

    LABEL_DICT = {  # help in mapping our class ids into TLA's ones
        "Lead": 0,
        "Position": 1,
        "Claim": 2,
        "Evidence": 3,
        "Counterclaim": 4,
        "Rebuttal": 5,
        "Concluding Statement": 6,
    }


def make_default_sub(df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Help to predict a default span for the very scarce essays with no predicted spans.

    Parameters
    ----------
    df_test: pd.DataFrame
        The dataframe containing the essays for which no spans were predicted by the models.

    Returns
    -------
    sub: pd.DataFrame
        A datafreme in the same format as the expected final output of this competition.
    """

    sub = df_test[["essay_id_comp"]].copy()
    sub["start"] = 0
    sub["end"] = df_test["full_text"].str.split().str.len()

    sub["predictionstring"] = sub["end"].apply(lambda n: " ".join(map(str, range(n))))

    sub["score_discourse_effectiveness_0"] = 0.99
    sub["score_discourse_effectiveness_1"] = 1 - sub["score_discourse_effectiveness_0"]
    sub["discourse_type"] = 0

    return sub


def df_to_buckets(df: pd.DataFrame, nbucket: int) -> List[Tuple[int, pd.DataFrame]]:
    """
    Split the DataFrame into nbucket sub-dataframes with approximately same size.
    """
    df_list = []
    sel = np.arange(len(df)) % nbucket
    for i in range(nbucket):
        idx = np.where(sel == i)[0]
        df_list.append((idx, df.iloc[idx]))

    return df_list


def read_df() -> pd.DataFrame:
    """
    Read the test dataframe form Infence config paths.
    """

    df = pd.read_csv(
        InfConfig.TEST_CSV_PATH,
        nrows=InfConfig.NUM_DEBUG_ROWS if InfConfig.IS_DEBUG else None,
    )

    df["num_words"] = df["full_text"].str.split().str.len()
    df.sort_values(["num_words"], ascending=False, inplace=True)

    logger.info(map_join(map_join("test_df.shape:", df.shape)))
    logger.info(map_join("uuids:", df["essay_id_comp"].iloc[:10].tolist()))

    return df


def get_params(fp16=False) -> List[Dict[str, Any]]:
    """
    Complete the specified model params, will also add tokenizer and model_config.
    """
    params = [
        inference.get_params(fp16=fp16, **param) for param in InfConfig.MODEL_PARAMS
    ]

    S = sum([param["weight"] for param in params])
    assert (
        abs(S - 1.0) < 1e-6
    ), f"The sum of all model weights must be 1.0 but {S:.3f} found"

    logger.info(
        map_join(
            [
                (iparam, param["model_name"], len(param["model_paths"]))
                for iparam, param in enumerate(params)
            ]
        )
    )

    return params


def predict_from_params_list(
    uuids: List[str],
    params: List[Dict[str, Any]],
    texts: Dict[str, str] = None,
    fp16=False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:
    """
    Get the predictions from the specified list of model params.
    """
    subs = []
    model_weights = []

    preds_eff = None

    for param in params:
        logger.info(
            map_join(
                "{}: {}".format(
                    Path(param["model_paths"][0]).stem, len(param["model_paths"])
                )
            )
        )
        preds, preds_seg, preds_eff_ = inference.predict_from_param(
            uuids=uuids,
            param=param,
            texts=texts,
            make_sub=False,
            oof=False,
            model_bar=True,
            fp16=fp16,
        )

        # The final segmentation has two components:
        # the first component is from direct segments predictions ==> weighted 0.6
        # the second compontent is from class predictions ==> weighted 0.4
        preds_seg = 0.60 * preds_seg + 0.40 * get_seg_from_ner(preds)

        subs.append(
            inference.make_sub_from_res(
                uuids=uuids, res=preds, res_seg=preds_seg, q=0.015, threshs=None
            )
        )
        preds_eff_ *= param["weight"]

        if preds_eff is None:
            preds_eff = preds_eff_
        else:
            preds_eff += preds_eff_

        model_weights.append(param["weight"])

        logger.info(map_join(subs[-1].shape))

    logger.info(map_join("model_weights:", model_weights))

    if not InfConfig.IS_DEBUG:
        del preds, preds_seg
        gc.collect()

    return preds_eff, subs, model_weights


def remove_noisy_spans(
    submit_df: pd.DataFrame,
    min_len_thresh: Dict[str, int],
    proba_thresh: Dict[str, float],
    use=("length", "probability"),
) -> pd.DataFrame:
    """
    Removes noisy spans using some common-sense rules:

    min_len_thresh ==> a minimum lenght constraint for each detected sapn
    proba_thresh ==> a minimum proba constraint for each detected sapn
    """
    df = submit_df.copy()
    df = df.fillna("")

    if "length" in use:
        df["l"] = df["predictionstring"].apply(lambda x: len(x.split()))
        for key, value in min_len_thresh.items():
            index = df.loc[df["class"] == key].query("l<%d" % value).index
            df.drop(index, inplace=True)

    if "probability" in use:
        for key, value in proba_thresh.items():
            index = df.loc[df["class"] == key].query("score<%f" % value).index
            df.drop(index, inplace=True)
    return df


def get_row_eff_score(
    row: pd.Series, preds_eff: Dict[str, pd.DataFrame], uuids_map: Dict[str, int]
) -> np.ndarray:
    """
    Compute the efficiency score for a given detected span.

    Parameters
    ----------
    row: pd.Series
        a row associated to a detected essay-span

    preds_eff:  a dictionary  of the associated efficiency prediction dataframe for each uuid

    uuids_map: a dict that maps each essay_id to its integer rank according to the Pytorch dataset order.
    """
    eff = (
        preds_eff[uuids_map[row["id"]]]
        .loc[row["start"] : (row["end"] - 1)]
        .mean()
        .values
    )
    return eff


def get_df_eff_score(
    df, preds_eff: pd.DataFrame, uuids_map: Dict[str, int], idx: list = None
) -> Tuple[list, np.ndarray]:
    """
    Compute the efficiency score for the detected spans in ``df``.
    """
    if idx is not None:
        assert len(idx) == len(df)

    try:  # depending on pandas versions, indexing a multi-index df
        # with duplicated keys can lead to an exception
        preds_eff = preds_eff.loc[[uuids_map[uuid] for uuid in df["id"].unique()]]
    except Exception as e:
        v = [uuids_map[uuid] for uuid in df["id"]]
        logger.info(map_join(len(set(df["id"])), len(set(v)), len(v)))
        logger.info(map_join(df.loc[df["id"].duplicated(False)]))
        raise e

    # Do .loc[i] to force groupby() level removal
    preds_eff = {i: tp.loc[i] for i, tp in preds_eff.groupby(level=0)}

    eff = np.c_[
        [
            get_row_eff_score(row, preds_eff=preds_eff, uuids_map=uuids_map)
            for _, row in df.iterrows()
        ]
    ]
    return idx, eff


def mp_get_eff_score(
    df: pd.DataFrame, preds_eff: pd.DataFrame, uuids_map: Dict[str, int]
) -> np.ndarray:
    """
    Compute the efficiency score for the detected spans in ``df`` in parallel.

    Returns
    ------
    eff: np.ndarray
        A numpy arrray of shape (len(df), 2) containing the efficiency scores.
    """

    T0 = time.time()

    df_list = df_to_buckets(df=df, nbucket=N_CPU)

    mapper = joblib.delayed(get_df_eff_score)
    tasks = [
        mapper(df=df_, preds_eff=preds_eff, uuids_map=uuids_map, idx=idx)
        for idx, df_ in df_list
        if len(df_)
    ]

    res = joblib.Parallel(N_CPU)(tasks)

    eff = np.zeros((len(df), preds_eff.shape[1]), dtype=np.float32)

    for idx, r in res:
        assert len(idx) == len(r)
        eff[idx] = r

    print_duration(time.time() - T0, desc="MP EFF SCORE DURATION")

    return eff


def finalize_subs(
    uuids: List[str],
    preds_eff: pd.DataFrame,
    subs: List[pd.DataFrame],
    weights: List[float],
) -> pd.DataFrame:
    """
    Apply the box fusion ensembling technique and remove noisy spans. It also computes efficiency scores.

    Parameters
    ----------
    uuids: List[str]
        The list of the essay ids
    preds_eff: pd.DataFrame
        Predicted efficiency scores from which the efficiency score for each span will  be derived

    subs: List[pd.DataFrame]
        A list of size M, containing predicted output by each model where M = total number of inference models.
        These prediction will combined using box fusion technique.

    Returns
    ------
    sub: pd.DataFrame
        A dataframe with the same format as the final output of this competition.
    """

    sub = fusion_boxes_for_subs(
        subs, weights, iou_thr=InfConfig.IOU_THR, skip_box_thr=InfConfig.SKIP_BOX_THR
    )

    sub = remove_noisy_spans(
        sub.reset_index(drop=True),
        InfConfig.MIN_LEN_THRESH_FOR_WBF,
        InfConfig.PROBA_THRESH_FOR_WBF,
        use=["probability"],
    )

    uuids_map = dict(zip(uuids, range(len(uuids))))

    # eff = np.c_[
    #     [
    #         get_row_eff_score(row, preds_eff=preds_eff, uuids_map=uuids_map)
    #         for _, row in tqdm(sub.iterrows(), total=len(sub), desc="EFFECTIVENESS")
    #     ]
    # ]

    # _, eff = get_df_eff_score(df=sub, preds_eff=preds_eff, uuids_map=uuids_map)

    eff = mp_get_eff_score(df=sub, preds_eff=preds_eff, uuids_map=uuids_map)

    sub[["score_discourse_effectiveness_0", "score_discourse_effectiveness_1"]] = eff

    return sub


def predict(
    config_yaml_path: Union[str, Path] = None,
    eff_bin_th: float | None = 0.40,
    use_dummy_eff: bool = False,
    ensure_complement: bool = True,
    fp16: bool = False,
):
    """
    Predict the final outputs using provided parameters in config_yaml_path.

    Please set eff_bin_th to None for raw effectiveness scores (no binarization).

    Parameters
    ----------

    config_yaml_path: str Path-like
        YAML config file containing everything
    eff_bin_th: float or None, (0, 1)
        A threshold used to binarize the effectiveness score, **score_discourse_effectiveness_0** will be set
        to **0.99** for all its values higher than this threshold and 0.01 otherwise, then **score_discourse_effectiveness_1**
        will be set to **1 - score_discourse_effectiveness_0** by complementarity.
        Please set eff_bin_th to None for raw effectiveness scores (no binarization).

    use_dummy_eff: bool, default=False
        Wether to simply ignore all the effectiveness score predictions or not. If set to True, then **score_discourse_effectiveness_0** will be set to
        0.99 and then   **score_discourse_effectiveness_1** will be set to **1 - score_discourse_effectiveness_0** by complementarity.

    ensure_complement: bool, default=True
        Wether to apply complementarity rule between **score_discourse_effectiveness_0** and **score_discourse_effectiveness_1** or not. For recall,
        complementarity implies that score_discourse_effectiveness_0 = 1 - score_discourse_effectiveness_1.

    fp16: bool, default=16
        Wether to use fp16 for faster inference but less accurate inference or not.
    """
    T0 = time.time()

    read_and_set_config(config_yaml_path, cfg=InfConfig)

    configs.LOGS_TXT_PATH = InfConfig.LOGS_TXT_PATH
    configs.DEVICE = InfConfig.DEVICE

    configs.init_config()

    assert eff_bin_th is None or not use_dummy_eff

    params = get_params(fp16=fp16)

    df = read_df()
    uuids = df["essay_id_comp"].tolist()
    texts = dict(zip(df["essay_id_comp"], df["full_text"]))

    preds_eff, subs, weights = predict_from_params_list(
        uuids=uuids, texts=texts, params=params, fp16=fp16
    )
    sub = finalize_subs(uuids=uuids, preds_eff=preds_eff, subs=subs, weights=weights)

    sub["essay_id_comp"] = sub["id"]
    sub["discourse_type"] = sub["class"].map(InfConfig.LABEL_DICT)

    if eff_bin_th is not None:
        bools = sub["score_discourse_effectiveness_0"] > eff_bin_th
        sub.loc[bools, "score_discourse_effectiveness_0"] = 0.99
        sub.loc[~bools, "score_discourse_effectiveness_0"] = 0.01

    elif use_dummy_eff:
        sub["score_discourse_effectiveness_0"] = 0.99

    if ensure_complement:
        sub["score_discourse_effectiveness_1"] = (
            1 - sub["score_discourse_effectiveness_0"]
        )

    sub = pd.concat(
        [
            sub,
            make_default_sub(
                df[~df["essay_id_comp"].isin(sub["essay_id_comp"])].copy()
            ),
        ],
        axis=0,
        sort=False,
        ignore_index=True,
    )

    sub[
        [
            "essay_id_comp",
            "predictionstring",
            "score_discourse_effectiveness_0",
            "score_discourse_effectiveness_1",
            "discourse_type",
        ]
    ].to_csv(InfConfig.SUB_CSV_SAVE_PATH, index=False)

    logger.info(f"Prediction CSV File Saved at: ``{InfConfig.SUB_CSV_SAVE_PATH}``")

    duration = int((time.time() - T0))

    print_duration(duration, desc="Global Run Duration")

    return sub


def main() -> None:
    """
    Main entry point for the RATER Challenge Model Inference Module.

    Parse command line arguments and initiate the prediction process.
    """
    import argparse

    parser = argparse.ArgumentParser(description="The RATER Challenge Inference Module")

    parser.add_argument(
        "--config_yaml_path",
        type=str,
        required=True,
        help="Path to the config YAML file.",
    )

    args = parser.parse_args()
    logger.info(map_join("CMD args:", args))
    predict(args.config_yaml_path)


if __name__ == "__main__":

    main()
