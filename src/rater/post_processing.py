import sys
from collections import Counter
from itertools import chain as it_chain
from typing import Any, Callable, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
import pandas as pd

from . import configs as cfg
from .configs import to_1_7

sys.setrecursionlimit(2048)

default_threshs = {
    "span_min_score": {
        "Claim": 0.43,
        "Concluding Statement": 0.44,
        "Counterclaim": 0.37,
        "Evidence": 0.42,
        "Lead": 0.38,
        "Position": 0.38,
        "Rebuttal": 0.36,
    },
    "span_min_len": {
        "Claim": 3,
        "Concluding Statement": 10,
        "Counterclaim": 7,
        "Evidence": 12,
        "Lead": 8,
        "Position": 5,
        "Rebuttal": 5,
    },
    "consecutive_span_min_score": {
        "Claim": 0.45,
        "Concluding Statement": float("+Inf"),
        "Counterclaim": float("+Inf"),
        "Evidence": 0.45,
        "Lead": float("+Inf"),
        "Position": float("+Inf"),
        "Rebuttal": float("+Inf"),
    },
    "consecutive_span_min_len": {
        "Claim": 3,
        "Concluding Statement": float("+Inf"),
        "Counterclaim": float("+Inf"),
        "Evidence": 14,
        "Lead": float("+Inf"),
        "Position": float("+Inf"),
        "Rebuttal": float("+Inf"),
    },
}


def get_most_common(L: List[Any]) -> Any:
    """
    Get the most common element in a list.

    Parameters
    ----------
    L : List[Any]
        The input list from which the most common element is to be found.

    Returns
    -------
    Any
        The most common element in the list.
    """
    return Counter(L).most_common(1)[0][0]


def get_seg_from_ner(res_ner: pd.DataFrame) -> pd.DataFrame:
    """
    Transform NER results into segmentation results.

    Parameters
    ----------
    res_ner : pd.DataFrame
        The NER results to be transformed.

    Returns
    -------
    pd.DataFrame
        The transformed segmentation results.
    """
    # Create a shallow copy of the input DataFrame
    res_seg = res_ner.copy(deep=False)

    # Get the columns of the DataFrame
    cols = res_seg.columns

    # Assign new column labels based on conditions
    cols = (
        1 * ((1 <= cols) & (cols <= cfg.NUM_PURE_TARGETS))  # In
        + 2
        * (
            (cfg.NUM_PURE_TARGETS < cols) & (cols <= 2 * cfg.NUM_PURE_TARGETS)
        )  # Beginning
        + 1
        * ((2 * cfg.NUM_PURE_TARGETS < cols) & (cols <= 3 * cfg.NUM_PURE_TARGETS))  # In
    )

    # Set new column labels
    res_seg.columns = cols

    # Group by the new column labels and sum within groups
    res_seg = res_seg.groupby(level=0, axis=1).sum()

    return res_seg


class SpanGetter:
    """
    Class to identify spans based on segmentation targets and optional prediction targets.

    Attributes
    ----------
    seg_target : np.ndarray
        Array of segmentation targets.
    pred_target : np.ndarray, optional
        Array of prediction targets.
    pred_target_argmax : List[int], optional
        List of argmax values from prediction targets.
    step : int
        Step size for moving through the targets.
    max_span_len : float
        Maximum length of a span.
    min_target_score : float
        Minimum score to consider for targets.
    start_checking_window : int
        Window size for checking start of span.
    end_checking_window : int
        Window size for checking end of span.
    nrows : int
        Number of rows in seg_target.
    """

    def __init__(
        self,
        seg_target: np.ndarray,
        pred_target: Optional[np.ndarray] = None,
        step: int = 3,
        max_span_len: Optional[int] = None,
        min_target_score: float = 0,
        start_checking_window: int = 3,
        end_checking_window: int = 1,
    ) -> None:
        """
        Initialize the SpanGetter.

        Parameters
        ----------
        seg_target : np.ndarray
            Array of segmentation targets.
        pred_target : np.ndarray, optional
            Array of prediction targets.
        step : int, default=3
            Step size for moving through the targets.
        max_span_len : int, optional
            Maximum length of a span.
        min_target_score : float, default=0
            Minimum score to consider for targets.
        start_checking_window : int, default=3
            Window size for checking start of span.
        end_checking_window : int, default=1
            Window size for checking end of span.
        """
        assert step > 0

        if pred_target is not None:
            assert len(seg_target) == len(pred_target)

        self.seg_target = np.asarray(seg_target).astype(np.float32, copy=False)
        self.pred_target = (
            np.asarray(pred_target).astype(np.float32, copy=False)
            if pred_target is not None
            else None
        )
        self.pred_target_argmax = (
            None
            if self.pred_target is None
            else [to_1_7(i) for i in self.pred_target.argmax(1)]
        )

        assert self.seg_target.shape[1] == 3
        assert self.seg_target.ndim == 2

        self.step = step
        self.max_span_len = float("+Inf") if max_span_len is None else max_span_len
        self.min_target_score = min_target_score
        self.start_checking_window = start_checking_window
        self.end_checking_window = end_checking_window
        self.nrows = len(seg_target)

    def find_next_start(self, pos: int, previous_end: int) -> Optional[int]:
        """
        Find the next start position for a span.

        Parameters
        ----------
        pos : int
            Current position in the target array.
        previous_end : int
            Previous end position of a span.

        Returns
        -------
        int, optional
            The next start position if found, else None.
        """
        if pos >= self.nrows - 1:
            return None

        while (pos < self.nrows) and (self.seg_target[pos].argmax() == 0):
            pos += 1

        if self.could_be_start(pos):
            return pos

        return self.find_next_start(pos=pos + 1, previous_end=previous_end)

    def find_next_end(self, pos: int, previous_start: int) -> int:
        """
        Find the next end position for a span.

        Parameters
        ----------
        pos : int
            Current position in the target array.
        previous_start : int
            Previous start position of a span.

        Returns
        -------
        int
            The next end position.
        """
        assert pos >= previous_start

        if pos >= self.nrows:
            return self.nrows

        if self.could_be_end(pos):
            return pos

        if (pos - previous_start) >= self.max_span_len:
            return previous_start + self.max_span_len

        return self.find_next_end(pos=pos + 1, previous_start=previous_start)

    def could_be_start(self, i: int) -> bool:
        """
        Check if the position could be the start of a span.

        Parameters
        ----------
        i : int
            Position in the target array.

        Returns
        -------
        bool
            True if it could be the start, else False.
        """
        if i >= self.nrows - 1:
            return False

        score = self.seg_target[i : i + self.start_checking_window].mean(0)
        return score.argmax() in [1, 2]

    def could_be_end(self, i: int) -> bool:
        """
        Check if the position could be the end of a span.

        Parameters
        ----------
        i : int
            Position in the target array.

        Returns
        -------
        bool
            True if it could be the end, else False.
        """
        if i >= self.nrows:
            return True

        score = self.seg_target[max(0, i - self.end_checking_window + 1) : i + 1].mean(
            0
        )
        return score.argmax() in [0, 2]

    def one_span(self, start: int = 0) -> Optional[dict]:
        """
        Identify one span starting from a given position.

        Parameters
        ----------
        start : int, default=0
            Starting position to identify the span.

        Returns
        -------
        dict, optional
            Dictionary containing span details or None if no span is found.
        """
        start = self.find_next_start(pos=start, previous_end=start)
        if start is None:
            return None

        end = self.find_next_end(pos=start + self.step, previous_start=start)

        span = {
            "start": start,
            "end": end,
            "class_id": (
                -1
                if self.pred_target_argmax is None
                else get_most_common(self.pred_target_argmax[start:end])
            ),
            "num_tokens": end - start,
        }
        return span

    def all_spans(self) -> List[dict]:
        """
        Identify all spans in the target array.

        Returns
        -------
        List[dict]
            List of dictionaries containing details of all spans.
        """
        start = 0
        spans = []
        while start < self.nrows:
            span = self.one_span(start=start)
            if span is not None:
                spans.append(span)
                start = span["end"]
            else:
                break
        return spans


def get_spans_from_seg_v3(
    seg_target: pd.DataFrame, pred_target: pd.DataFrame = None
) -> List[dict]:
    """
    Get spans from segmentation target using SpanGetter.

    Parameters
    ----------
    seg_target : torch.Tensor
        The segmentation target tensor.
    pred_target : torch.Tensor, optional
        The predicted target tensor, by default None.

    Returns
    -------
    List[dict]
        A list of spans infos.
    """
    return SpanGetter(seg_target=seg_target, pred_target=pred_target).all_spans()


class SpanRepairer:
    """
    Class to repair spans based on specified criteria.
    """

    def __init__(
        self,
        span_min_len: dict,
        span_min_score: dict,
        consecutive_span_min_len: dict,
        consecutive_span_min_score: dict,
        max_iter: Optional[int] = None,
    ):
        """
        Initialize the SpanRepairer with given parameters.

        Parameters
        ----------
        span_min_len : dict
            Minimum length of the span for each class.
        span_min_score : dict
            Minimum score of the span for each class.
        consecutive_span_min_len : dict
            Minimum length of consecutive spans for each class.
        consecutive_span_min_score : dict
            Minimum score of consecutive spans for each class.
        max_iter : Optional[int], default=None
            Maximum iterations for the repair process.
        """
        self.span_min_len = span_min_len
        self.span_min_score = span_min_score
        self.consecutive_span_min_len = consecutive_span_min_len
        self.consecutive_span_min_score = consecutive_span_min_score

        self.max_iter = max_iter or 1_000

    @staticmethod
    def merge_spans(span1: dict, span2: dict, copy: bool = False) -> dict:
        """
        Merge two spans into one.

        Parameters
        ----------
        span1 : dict
            First span.
        span2 : dict
            Second span.
        copy : bool, default=False
            If True, create a copy of span1 before merging.

        Returns
        -------
        dict
            Merged span.
        """
        span = span1.copy() if copy else span1

        span["start"] = min(span1["start"], span2["start"])
        span["end"] = max(span1["end"], span2["end"])

        span["score"] = 0.5 * (span1["score"] + span2["score"])
        span["num_tokens"] = span["end"] - span["start"]

        return span

    def is_span_ok(self, span: dict) -> bool:
        """
        Check if a span meets the minimum length and score criteria.

        Parameters
        ----------
        span : dict
            Span to check.

        Returns
        -------
        bool
            True if span is valid, False otherwise.
        """
        return (span["num_tokens"] >= self.span_min_len[span["class"]]) and (
            span["score"] >= self.span_min_score[span["class"]]
        )

    def is_consecutive_spans_ok(self, span: dict, last_span: dict) -> bool:
        """
        Check if two consecutive spans meet the criteria.

        Parameters
        ----------
        span : dict
            Current span.
        last_span : dict
            Previous span.

        Returns
        -------
        bool
            True if consecutive spans are valid, False otherwise.
        """
        assert span["class"] == last_span["class"]

        return (
            (span["num_tokens"] >= self.consecutive_span_min_len[span["class"]])
            and (
                last_span["num_tokens"]
                >= self.consecutive_span_min_len[last_span["class"]]
            )
            and (span["score"] >= self.consecutive_span_min_score[span["class"]])
            and (
                last_span["score"]
                >= self.consecutive_span_min_score[last_span["class"]]
            )
        )

    def move_cusrsor(
        self, span1: dict, span2: dict, copy: bool = False
    ) -> Tuple[List[dict], int]:
        """
        Move cursor and decide whether to merge spans or not.

        Parameters
        ----------
        span1 : dict
            First span.
        span2 : dict
            Second span.
        copy : bool, default=False
            If True, create a copy before merging.

        Returns
        -------
        Tuple[Tuple[dict, ...], int]
            Tuple containing spans to append and the number of steps.
        """
        if span1["class"] == span2["class"]:
            if (
                self.is_span_ok(span1)
                and self.is_span_ok(span2)
                and self.is_consecutive_spans_ok(span1, span2)
            ):
                spans, num_step = (span1,), 1

            else:
                spans, num_step = (self.merge_spans(span1, span2, copy=copy),), 2
        else:
            if self.is_span_ok(span1):
                spans, num_step = (span1,), 1
            else:
                spans, num_step = (), 1

        return spans, num_step

    def smart_repair(
        self,
        spans: List[dict],
        res_list: Optional[List[dict]] = None,
        copy: bool = False,
    ) -> List[dict]:
        """
        Repair spans smartly by merging or discarding based on criteria.

        Parameters
        ----------
        spans : List[dict]
            List of spans to repair.
        res_list : Optional[List[dict]], default=None
            Result list to append repaired spans.
        copy : bool, default=False
            If True, create copies during the process.

        Returns
        -------
        List[dict]
            Repaired list of spans.
        """
        if res_list is None:
            res_list = []

        if len(spans) == 0:
            return res_list

        if len(spans) == 1:
            if self.is_span_ok(spans[0]):
                res_list.append(spans[0])

            return res_list

        span1, span2 = spans[0], spans[1]

        spans_to_append, num_step = self.move_cusrsor(span1, span2, copy=copy)

        assert num_step > 0

        res_list.extend(spans_to_append)

        for _ in range(num_step):
            spans.pop(0)

        return self.smart_repair(spans, res_list=res_list, copy=copy)

    def __call__(self, spans: List[dict], copy: bool = False) -> List[dict]:
        """
        Call method to perform the repair process on spans.

        Parameters
        ----------
        spans : List[dict]
            List of spans to repair.
        copy : bool, default=False
            If True, create copies during the process.

        Returns
        -------
        List[dict]
            Repaired list of spans.
        """
        max_iter = self.max_iter or float("+Inf")
        i = 0

        while i < max_iter:
            n = len(spans)
            spans = self.smart_repair(spans, copy=copy)

            if len(spans) >= n:
                break

            i += 1

        return spans

    def repair_consecutive_spans(
        self, spans: List[dict], copy: bool = False
    ) -> List[dict]:
        """
        Repair consecutive spans based on criteria.

        Parameters
        ----------
        spans : List[dict]
            List of spans to repair.
        copy : bool, default=False
            If True, create copies during the process.

        Returns
        -------
        List[dict]
            Repaired list of consecutive spans.
        """
        if len(spans) < 2:
            return spans

        last_span = spans[0]
        new_spans = [last_span]
        for i, span in enumerate(spans[1:], 1):
            if span["class"] == last_span["class"]:
                if self.is_consecutive_spans_ok(span, last_span):
                    new_spans.append(span)

                else:
                    span = self.merge_spans(span, last_span, copy=copy)
                    new_spans[-1] = span
            else:
                new_spans.append(span)

            last_span = span

        return new_spans

    def repair_single_spans(self, spans: List[dict], copy: bool = False) -> List[dict]:
        """
        Repair individual spans based on criteria.

        Parameters
        ----------
        spans : List[dict]
            List of spans to repair.
        copy : bool, default=False
            If True, create copies during the process.

        Returns
        -------
        List[dict]
            Repaired list of single spans.
        """
        if copy:
            spans = spans.copy()

        for i in range(len(spans) - 1, -1, -1):
            span = spans[i]

            if not self.is_span_ok(span):
                spans.pop(i)

        return spans


def get_targets_from_spans_v2(
    spans: List[Dict[str, Any]], preds: np.ndarray, zero_margin: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Process spans and predictions to determine target classes and scores.

    Parameters
    ----------
    spans : List[Dict[str, Union[int, str]]]
        List of spans, each containing 'start' and 'end' keys.
    preds : torch.Tensor
        Tensor containing prediction scores.
    zero_margin : float, optional
        Margin to add to the zero class score, by default 0.20.

    Returns
    -------
    List[Dict[str, Union[int, str, float]]]
        List of spans with added 'score', 'class', and 'class_id' keys if the score exceeds the zero class score plus margin.
    """
    new_spans = []
    for span in spans:
        scores = preds[span["start"] : span["end"]].mean(0)
        score_zero = scores[0] + zero_margin

        class_id = scores.argmax()
        score = scores[class_id]
        if score > score_zero:
            span["score"] = score
            class_id = to_1_7(class_id)
            span["class"] = cfg.ID2DISCOURSE[class_id] if class_id > 0 else ""
            span["class_id"] = class_id

            new_spans.append(span)

    return new_spans


def get_thresh_from_sub(
    sub: pd.DataFrame,
    q: float = None,
    q_consecutive: float = None,
) -> dict:
    """
    Compute threshold values for different metrics based on the quantiles of the provided dataframe.

    Parameters
    ----------
    sub : pd.DataFrame
        Input dataframe containing columns 'class', 'score', and 'num_tokens'.
    q : float, optional
        Quantile value for single spans.
    q_consecutive : float, optional
        Quantile value for consecutive spans. If not provided, defaults to 1.5 times `q`.

    Returns
    -------
    dict
        Dictionary containing threshold values for span scores and lengths,
        both for single and consecutive spans.
    """
    if q_consecutive is None:
        q_consecutive = 1.5 * q

    span_min_score = sub.groupby("class")["score"].quantile(q).round(2).to_dict()
    span_min_len = sub.groupby("class")["num_tokens"].quantile(q).astype(int).to_dict()

    consecutive_span_min_score = (
        sub.groupby("class")["score"]
        .quantile(q_consecutive, interpolation="nearest")
        .round(2)
        .to_dict()
    )
    consecutive_span_min_len = (
        sub.groupby("class")["num_tokens"]
        .quantile(q_consecutive, interpolation="nearest")
        .astype(int)
        .to_dict()
    )

    threshs = {
        "span_min_score": span_min_score,
        "span_min_len": span_min_len,
        "consecutive_span_min_score": consecutive_span_min_score,
        "consecutive_span_min_len": consecutive_span_min_len,
    }

    # For non-constrained classes, set thresholds to +Inf
    for thresh_name in threshs:
        for class_ in cfg.DISCOURSE2ID:
            threshs[thresh_name][class_] = threshs[thresh_name].get(
                class_, float("+Inf")
            )

            if (
                thresh_name
                in ["consecutive_span_min_score", "consecutive_span_min_len"]
            ) and (class_ not in ["Evidence", "Claim"]):
                threshs[thresh_name][class_] = float("+Inf")

    return threshs


def get_preds_seg_v2(
    uuids: List[str],
    res: pd.DataFrame,
    res_seg: pd.DataFrame,
    span_getter: Callable = None,
    target_getter: Callable = None,
    span_getter_kwargs: dict = None,
    target_getter_kwargs: dict = None,
) -> Dict[str, List[Dict]]:
    """
    Generate predictions for segments.

    Parameters
    ----------
    uuids : List[str]
        List of unique identifiers.
    res : pd.DataFrame
        DataFrame containing the results.
    res_seg : pd.DataFrame
        DataFrame containing the segment results.
    span_getter : Callable, optional
        Function to get spans from segments, by default get_spans_from_seg_v3.
    target_getter : Callable, optional
        Function to get targets from spans, by default get_targets_from_spans_v2.
    span_getter_kwargs : dict, optional
        Additional arguments for span_getter function, by default None.
    target_getter_kwargs : dict, optional
        Additional arguments for target_getter function, by default None.

    Returns
    -------
    Dict[str, List[Dict]]
        Dictionary mapping uuids to lists of spans with predictions.
    """
    span_getter = get_spans_from_seg_v3 if span_getter is None else span_getter
    target_getter = (
        get_targets_from_spans_v2 if target_getter is None else target_getter
    )

    span_getter_kwargs = span_getter_kwargs or {}
    target_getter_kwargs = target_getter_kwargs or {}

    all_spans = {}

    # Copy and transform the results DataFrame
    res = res.copy()
    res.columns = [to_1_7(col) for col in res.columns]
    res = res.groupby(level=0, axis=1).sum()

    for index, df_uuid in res.groupby(level=0):

        seg_target = res_seg.loc[index].values
        spans = span_getter(seg_target, pred_target=None, **span_getter_kwargs)

        spans = target_getter(spans, preds=df_uuid.values, **target_getter_kwargs)

        id_ = uuids[df_uuid.index[0][0]]
        for i in range(
            len(spans) - 1, -1, -1
        ):  # In reverse order to avoid popping issues
            span = spans[i]
            if span["class_id"] <= 0:
                spans.pop(i)
            else:
                span["id"] = id_

        all_spans[id_] = spans

    return all_spans


def prune_spans(
    spans: List[dict], threshs: Dict[str, float], max_iter: int = None
) -> List[dict]:
    """
    Prune spans based on given thresholds.

    Parameters
    ----------
    spans : List[dict]
        List of spans to be pruned.
    threshs : Dict[str, Any]
        Dictionary of thresholds for pruning.
    max_iter : int, optional
        Maximum number of iterations for pruning, by default None

    Returns
    -------
    List[Tuple[int, int]]
        List of pruned spans.
    """
    repairer = SpanRepairer(**threshs, max_iter=max_iter)
    return repairer(spans)


def add_predictionstring_to_sub(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'predictionstring' column to the DataFrame 'sub'.

    Parameters
    ----------
    sub : pd.DataFrame
        DataFrame containing 'start' and 'end' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'predictionstring' column.
    """
    sub["predictionstring"] = sub[["start", "end"]].apply(
        lambda span: " ".join(map(str, range(span["start"], span["end"]))),
        axis=1,
    )
    return sub


def make_sub_from_res(
    uuids: List[str],
    res: np.ndarray,
    res_seg: np.ndarray,
    span_getter: Callable = None,
    threshs: Optional[Dict[str, float]] = None,
    q: float = 0.015,
    max_iter: Optional[int] = None,
    prune: bool = True,
) -> pd.DataFrame:
    """
    Generate submission DataFrame from prediction results.

    Parameters
    ----------
    uuids : List[str]
        List of unique identifiers for the data samples.
    res : np.ndarray
        Array containing the prediction results.
    res_seg : np.ndarray
        Array containing the segmentation results.
    span_getter : Callable, optional
        Function to get spans from results, by default None.
    threshs : Optional[Dict[str, float]], optional
        Dictionary of thresholds for pruning, by default None.
    q : float, optional
        Quantile value for threshold calculation, by default 0.015.
    max_iter : Optional[int], optional
        Maximum iterations for pruning, by default None.
    prune : bool, optional
        Flag to determine whether to prune spans, by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the submission data.
    """

    # Get spans from the results
    spans = get_preds_seg_v2(
        uuids=uuids, res=res, res_seg=res_seg, span_getter=span_getter
    )

    # Convert spans to a DataFrame
    sub = pd.DataFrame(list(it_chain(*spans.values())))

    if not len(sub):
        return sub

    # If no pruning, add prediction string to the submission and return
    if not prune:
        sub = add_predictionstring_to_sub(sub)
        return sub

    # Calculate thresholds if not provided
    if threshs is None:
        threshs = get_thresh_from_sub(sub, q=q)

    # Prune spans based on thresholds
    new_spans = {}
    for uuid, spans_ in spans.items():
        new_spans[uuid] = prune_spans(spans=spans_, threshs=threshs, max_iter=max_iter)

    new_spans = list(it_chain(*new_spans.values()))

    # Warn if all spans were filtered out and revert to initial spans
    if not len(new_spans):
        warn(
            "All spans were filtered out, so will remove span filtering and keep initial spans."
        )
        new_spans = list(it_chain(*spans.values()))

    sub = pd.DataFrame(new_spans)

    # Add prediction string to the submission if not empty
    if len(sub):
        sub = add_predictionstring_to_sub(sub)

    return sub
