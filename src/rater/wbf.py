from typing import Dict, List, Tuple
from warnings import warn

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from tqdm.auto import tqdm

from . import configs as cfg


def to_rectangle(box_seg: np.ndarray, norm: float = None) -> np.ndarray:
    """
    Convert segmented box coordinates to rectangular coordinates.

    Parameters
    ----------
    box_seg : np.ndarray
        Segmented box coordinates.
    norm : float, optional
        Normalization factor, by default None.

    Returns
    -------
    np.ndarray
        Rectangular box coordinates.
    """
    box_rect = np.zeros((len(box_seg), 4), dtype=np.float32)
    box_rect[:, [0, 2]] = box_seg

    if norm:
        box_rect /= norm

    box_rect[:, -1] = 1

    return box_rect


def fusion_boxes_for_uuid(
    uuid: str,
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    weights: List[float],
    iou_thr: float = 0.333,
    skip_box_thr: float = 0.001,
    default_box_idx: int = None,
) -> pd.DataFrame:
    """
    Perform weighted boxes fusion for a given UUID and return a DataFrame with the fused results.

    Parameters
    ----------
    uuid : str
        Unique identifier for the set of boxes.
    boxes_list : List[np.ndarray]
        List of numpy arrays containing bounding boxes.
    scores_list : List[np.ndarray]
        List of numpy arrays containing scores for each box.
    labels_list : List[np.ndarray]
        List of numpy arrays containing labels for each box.
    weights : List[float]
        List of weights for each set of boxes.
    iou_thr : float, optional
        Intersection over Union (IoU) threshold for the fusion, by default 0.333.
    skip_box_thr : float, optional
        Threshold to skip boxes, by default 0.001.
    default_box_idx : int, optional
        Index of the default box to use if no boxes are left after fusion, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the fused boxes, scores, and labels.
    """

    Z = 1e-3 + max(map(np.max, boxes_list))

    if default_box_idx is None:
        default_box_idx = np.argmax(weights)

    new_boxes_list = [to_rectangle(box, norm=Z) for box in boxes_list]
    boxes_list = new_boxes_list

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    if not len(boxes):
        boxes, scores, labels = (
            boxes_list[default_box_idx],
            scores_list[default_box_idx],
            labels_list[default_box_idx],
        )

    boxes = (boxes * Z).round().astype(int)
    labels = labels.astype(int)

    sub = pd.DataFrame(
        {
            "id": uuid,
            "class_id": labels,
            "class": [cfg.ID2DISCOURSE[class_] for class_ in labels],
            "score": scores,
            "start": boxes[:, 0],
            "end": boxes[:, 2],
            "predictionstring": [
                " ".join(map(str, range(start, end)))
                for start, end in zip(boxes[:, 0], boxes[:, 2])
            ],
        }
    )

    sub = sub.sort_values(["start", "end"])

    return sub


def get_uuids(subs: List[Dict[str, List[str]]]) -> Tuple[List[str], List[str]]:
    """
    Extract and sort unique and common UUIDs from a list of subscriptions.

    Parameters
    ----------
    subs : List[Dict[str, List[str]]]
        A list of dictionaries, each containing an "id" key with a list of UUID strings.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists: all unique UUIDs and common UUIDs.
    """
    assert len(subs), "The list of subsmissions cannot be empty."

    uuids = set(subs[0]["id"])

    all_uuids = set(uuids)
    common_uuids = set(uuids)

    for sub in subs[1:]:
        uuids = set(sub["id"])

        all_uuids = all_uuids.union(uuids)
        common_uuids = common_uuids.intersection(uuids)

    common_uuids = sorted(common_uuids)
    all_uuids = sorted(all_uuids)

    return all_uuids, common_uuids


def fusion_boxes_for_subs(
    subs: List[pd.DataFrame],
    weights: np.ndarray = None,
    bar: bool = True,
    iou_thr: float = 0.333,
    skip_box_thr: float = 0.001,
    default_box_idx: int = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Fuses bounding boxes from multiple submissions based on the specified criteria.

    Parameters
    ----------
    subs : List[pd.DataFrame]
        List of dataframes containing bounding boxes.
    weights : np.ndarray, optional
        Weights for each dataframe, by default None
    bar : bool, optional
        Whether to display a progress bar, by default True
    iou_thr : float, optional
        Intersection over Union (IoU) threshold, by default 0.333
    skip_box_thr : float, optional
        Score threshold to skip boxes, by default 0.001
    default_box_idx : int, optional
        Default box index, by default None
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    pd.DataFrame
        Dataframe with fused bounding boxes
    """

    assert len(subs)

    if weights is None:
        weights = np.ones(len(subs))
        weights /= weights.sum()

    all_uuids, common_uuids = get_uuids(subs)

    other_uuids = sorted(set(all_uuids).difference(common_uuids))

    if len(other_uuids):
        warn(
            f"{len(other_uuids)} ids are not present in all the dataframes, this could be problematic !"
        )

    res = []

    if bar:
        all_uuids = tqdm(all_uuids, desc="BOX FUSION")

    for uuid in all_uuids:
        boxes_list, scores_list, labels_list = [], [], []
        valid_weights = []

        for w, sub in zip(weights, subs):
            temp = sub.query(f"id == '{uuid}'")

            if not len(temp):
                continue

            boxes_list.append(temp[["start", "end"]].values)
            scores_list.append(temp["score"].values)
            labels_list.append(temp["class_id"].values)
            valid_weights.append(w)

        sub = fusion_boxes_for_uuid(
            uuid=uuid,
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            weights=valid_weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            default_box_idx=default_box_idx,
            **kwargs,
        )

        res.append(sub)

    sub = pd.concat(res, axis=0, sort=False, ignore_index=True)
    sub.reset_index(drop=True, inplace=True)

    return sub
