import gc
import math
import os
from collections import defaultdict
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    get_cosine_schedule_with_warmup,
)

from . import configs as cfg
from .comp_metric import score_feedback_comp
from .configs import logger
from .dataset import (
    Dataset,
    DynamicBatchDataset,
    TestDataset,
    collate_fn_list,
    collate_fn_train,
    read_from_id,
)
from .inference import predict_eval
from .model_saving import AutoSave
from .models import (
    Model,
    NERSegmentationLoss,
    check_if_model_has_position_embeddings,
)
from .models import load_model_weights as models_load_model
from .post_processing import (
    default_threshs,
    get_seg_from_ner,
    make_sub_from_res,
)
from .utils import seed_everything, slugify

average = "macro"
multi_class = "ovo"  # "ovr"
zero_division = 0


def disable_tokenizer_parallelism():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_tokenizer(
    model_name: str, config: Optional[PretrainedConfig] = None
) -> Tuple[bool, PreTrainedTokenizer]:
    """
    Loads a tokenizer and adds special tokens.

    Parameters
    ----------
    model_name : str
        The name of the pre-trained model to load the tokenizer from.
    config : PretrainedConfig, optional
        Optional configuration for the tokenizer.

    Returns
    -------
    Tuple[bool, PreTrainedTokenizer]
        A tuple containing a boolean indicating if any tokens were added and the tokenizer instance.
    """
    # Load tokenizer with specified model name and configuration
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trim_offsets=False, config=config
    )

    # Add special tokens to the tokenizer
    num_added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["\t", "\n", "\r", "\x0c", "\x0b"]},
        replace_additional_special_tokens=False,
    )

    # Check if any tokens were added
    added_any_tokens = num_added_tokens > 0

    return added_any_tokens, tokenizer


def one_step(
    inputs: tuple,
    net: torch.nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    Perform one training step.

    Parameters
    ----------
    inputs :
        The input data for the model.
    net : torch.nn.Module
        The neural network model.
    criterion : Callable
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler], optional
        The learning rate scheduler, by default None.
    scaler : Optional[torch.cuda.amp.GradScaler], optional
        The gradient scaler for mixed precision training, by default None.

    Returns
    -------
    Dict[str, float]
        The metrics dictionary containing loss, accuracy, F1 score, precision, and recall.
    """
    if len(inputs) == 4:
        word_perc_pos, input_ids, attention_mask, target = inputs
    else:
        (
            batch_ids,
            word_ids,
            word_perc_pos,
            input_ids,
            attention_mask,
            target,
            eff_target,
        ) = inputs

    # Convert numpy arrays to torch tensors if necessary
    if isinstance(input_ids, np.ndarray):
        input_ids = torch.from_numpy(input_ids)
        attention_mask = torch.from_numpy(attention_mask)
        target = torch.from_numpy(target)
        eff_target = torch.from_numpy(eff_target)
        word_perc_pos = torch.from_numpy(word_perc_pos)

    # Move tensors to the device
    input_ids = input_ids.to(cfg.DEVICE)
    attention_mask = attention_mask.to(cfg.DEVICE)
    target = target.to(cfg.DEVICE)
    eff_target = eff_target.to(cfg.DEVICE)
    word_perc_pos = word_perc_pos.to(cfg.DEVICE)

    optimizer.zero_grad(set_to_none=True)

    use_amp = scaler is not None

    if use_amp:
        with torch.cuda.amp.autocast():
            # See model definition in models.Model for output format
            o = net(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_ids=word_perc_pos,
            )
            # See loss definition in models.NERSegmentationLoss for details on the loss
            loss = criterion(*o, target=target, eff_target=eff_target)
    else:
        # See model definition for output format
        o = net(
            input_ids=input_ids, attention_mask=attention_mask, word_ids=word_perc_pos
        )
        # See loss definition in models.NERSegmentationLoss for details on the loss
        loss = criterion(*o, target=target, eff_target=eff_target)

    if use_amp:
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.CLIP_GRAD_NORM)

        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        l_val = loss.item()
        if np.isnan(l_val):
            l_val = 0.0

        o = o[0].mean(1)
        o = o.argmax(-1).cpu().numpy().ravel()
        yb = target.cpu().numpy().ravel()

        sel = yb >= 0
        o = o[sel]
        yb = yb[sel]

        acc = accuracy_score(yb, o)
        prec = precision_score(yb, o, average=average, zero_division=zero_division)
        rec = recall_score(yb, o, average=average, zero_division=zero_division)
        f1 = f1_score(yb, o, average=average, zero_division=zero_division)

    if scheduler is not None:
        scheduler.step()

    metrics = {
        "loss": l_val,
        "acc": acc,
        "f1": f1,
        "prec": prec,
        "rec": rec,
    }

    return metrics


def get_metrics(out: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for model predictions.

    Parameters
    ----------
    out : np.ndarray
        The predicted output labels.
    target : np.ndarray
        The true target labels.

    Returns
    -------
    Dict[str, float]
        Dictionary containing accuracy, precision, recall, and F1 score.
    """
    # Calculate accuracy
    acc = accuracy_score(target, out)

    # Calculate precision, recall, and F1 score
    prec = precision_score(target, out, average=average, zero_division=zero_division)
    rec = recall_score(target, out, average=average, zero_division=zero_division)
    f1 = f1_score(target, out, average=average, zero_division=zero_division)

    # Store metrics in a dictionary
    metrics = {
        "acc": acc,
        "f1": f1,
        "prec": prec,
        "rec": rec,
    }

    return metrics


def get_macro_f1_score(
    uuids: List[str], df: pd.DataFrame, res: Dict[str, Any]
) -> Tuple[Tuple[float, Dict[str, float]], Tuple[float, Dict[str, float]]]:
    """
    Calculate the macro F1 score for given predictions.

    Parameters
    ----------
    uuids : List[str]
        List of unique identifiers.
    df : pd.DataFrame
        DataFrame containing the ground truth data.
    res : Dict[str, Any]
        Dictionary containing prediction results.

    Returns
    -------
    Tuple[Tuple[float, Dict[str, float]], Tuple[float, Dict[str, float]]]
        A tuple containing macro F1 score and class scores for both original and adjusted segmentations.
    """

    def return_default() -> (
        Tuple[Tuple[float, Dict[str, float]], Tuple[float, Dict[str, float]]]
    ):
        """
        Return default values in case of an error.

        Returns
        -------
        Tuple[Tuple[float, Dict[str, float]], Tuple[float, Dict[str, float]]]
            Default macro F1 score and class scores.
        """
        macrof1_score = -1.0
        class_scores = {}
        macrof1_score_v2 = -1.0
        class_scores_v2 = {}
        return (macrof1_score, class_scores), (macrof1_score_v2, class_scores_v2)

    preds = res["preds"]
    res_seg = res["preds_seg"]
    threshs = deepcopy(default_threshs)

    try:
        sub = make_sub_from_res(
            uuids=uuids, res=preds, res_seg=res_seg, threshs=threshs
        )

        res_seg_v2 = cfg.TRUE_SEG_COEF * res_seg + (
            1 - cfg.TRUE_SEG_COEF
        ) * get_seg_from_ner(preds)

        sub_v2 = make_sub_from_res(
            uuids=uuids, res=preds, res_seg=res_seg_v2, threshs=threshs
        )
    except Exception as e:
        warn(f"EXCEPTION DURING SUB BUILDING !!!\n{e}")
        return return_default()

    if len(sub):
        try:
            macrof1_score, class_scores = score_feedback_comp(
                sub, df, return_class_scores=True
            )

            macrof1_score_v2, class_scores_v2 = score_feedback_comp(
                sub_v2, df, return_class_scores=True
            )
        except Exception as e:
            warn(f"EXCEPTION DURING MACRO F1 COMPUTATION !!!\n{e}")
            return return_default()
    else:
        warn("empty discourse prediction, f1_score can't be computed !", UserWarning)
        return return_default()

    return (macrof1_score, class_scores), (macrof1_score_v2, class_scores_v2)


@torch.no_grad()
def evaluate(
    net: torch.nn.Module, val_loader: torch.utils.data.DataLoader, df: pd.DataFrame
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Evaluate the model on the validation dataset.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model to evaluate.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    df : pd.DataFrame
        DataFrame containing additional information for evaluation.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, float]]
        A tuple containing the out-of-fold predictions and metrics.
    """
    net.eval()

    uuids = val_loader.dataset.dataset.uuids
    df = df[df["id"].isin(uuids)].reset_index(drop=True)
    val_loader = tqdm(val_loader, leave=False, total=len(val_loader))

    res = predict_eval(net, val_loader, ret_out=False, dynamic_padding=True)

    preds = res["preds"]
    target_v2 = res["target_v2"]

    (macrof1_score, class_scores), (macrof1_score_v2, class_scores_v2) = (
        get_macro_f1_score(uuids, df=df, res=res)
    )

    l = -1.0  # noqa: E741

    oof = {
        "uuids": uuids,
        "out": preds.astype(np.float16),
        "out_seg": res["preds_seg"].astype(np.float16),
        "target": target_v2.astype(np.int8),
        "out_eff": res["preds_eff"],
        "eff_target": res["eff_target_v2"],
    }

    metrics = {}

    metrics.update(
        iov=macrof1_score,
        iov_classes=class_scores,
        iov_v2=macrof1_score_v2,
        iov_classes_v2=class_scores_v2,
        loss=l,
    )

    o_v2 = preds.values  # .argmax(1)
    target_v2 = target_v2.values[:, 0]

    o_v2 = o_v2.argmax(1)

    metrics.update(get_metrics(o_v2, target_v2))

    o_v2 = res["preds_eff"].values.argmax(1)
    target_v2 = res["eff_target_v2"].values[:, 0]
    bools = o_v2 >= 0
    m = get_metrics(o_v2[bools], target_v2[bools])

    metrics.update({f"{k}_eff": v for k, v in m.items()})

    return oof, metrics


def finalize_epoch(*, net, criterion, val_loader, df, icount, train_metrics, **kwargs):

    metrics = deepcopy(train_metrics)
    div_factor = max(1, icount)
    for key in list(metrics):
        metrics[key] /= div_factor

    oof, metrics_val = evaluate(net, val_loader, df=df)

    for key, val in metrics_val.items():
        metrics[f"{key}_val"] = val

    print_and_save(
        oof=oof,
        metrics=metrics,
        net=net,
        criterion=criterion,
        **kwargs,
    )

    return oof, metrics


def one_epoch(
    *,
    net: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    df: pd.DataFrame,
    schedule_each_step: bool = False,
    scaler: torch.cuda.amp.GradScaler = None,
    save_each: int = None,
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Perform one training epoch.

    Parameters
    ----------
    net : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for the network.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    df : pd.DataFrame
        DataFrame containing ground truths.
    schedule_each_step : bool, optional
        Whether to step the scheduler each iteration (default is False).
    scaler : torch.cuda.amp.GradScaler, optional
        Gradient scaler for mixed precision training (default is None).
    save_each : int, optional
        Save model every `save_each` epochs (default is None).
    kwargs : dict
        Additional arguments.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, float]]
        The out-of-fold predictions and the epoch metrics.
    """
    net.train()  # Set the network to training mode
    icount = 0
    num_iter = len(train_loader)
    train_loader = tqdm(train_loader, leave=False)
    epoch_bar = train_loader

    metrics_format_dict = dict(
        loss="{loss:.6f}",
        acc="{acc:.3f}",
        prec="{prec:.3f}",
        rec="{rec:.3f}",
        f1="{f1:.3f}",
    )

    if save_each is not None:
        save_step = save_each * num_iter
        save_points = np.arange(save_step, num_iter, save_step).astype(int)
    else:
        save_points = []

    metrics = defaultdict(int)

    for step, inputs in enumerate(epoch_bar):
        _metrics = one_step(
            inputs, net=net, criterion=criterion, optimizer=optimizer, scaler=scaler
        )

        if schedule_each_step:
            scheduler.step()

        for key, val in _metrics.items():
            metrics[key] += val

        icount += 1

        if hasattr(epoch_bar, "set_postfix") and not icount % 10:
            metrics_normalized = {key: val / icount for key, val in metrics.items()}
            metrics_formated = {
                key: val.format(**metrics_normalized)
                for key, val in metrics_format_dict.items()
            }

            epoch_bar.set_postfix(**metrics_formated)

        if step in save_points:
            oof, metrics_temp = finalize_epoch(
                net=net,
                criterion=criterion,
                val_loader=val_loader,
                df=df,
                icount=icount,
                train_metrics=metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                **kwargs,
            )

    if not schedule_each_step:
        scheduler.step()

    oof, metrics = finalize_epoch(
        net=net,
        criterion=criterion,
        val_loader=val_loader,
        df=df,
        icount=icount,
        train_metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        **kwargs,
    )

    return oof, metrics


def fetch_optimizer(net: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Fetches an AdamW optimizer for the given network.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network for which the optimizer is to be created.

    Returns
    -------
    torch.optim.Optimizer
        The AdamW optimizer configured with the network's parameters, learning rate, and weight decay.
    """
    return torch.optim.AdamW(
        net.parameters(), lr=cfg.OPTIMIZER_LR, weight_decay=cfg.OPTIMIZER_WEIGHT_DECAY
    )


def fetch_scheduler(
    optimizer: torch.optim.Optimizer, num_train_steps: int
) -> torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
    """
    Fetches a cosine scheduler with warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer for which to schedule the learning rate.
    num_train_steps : int
        The total number of training steps.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        The learning rate scheduler.
    """
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            0.1 * num_train_steps
        ),  # 10% of total steps used for warmup
        num_training_steps=num_train_steps,
        num_cycles=1,
        last_epoch=-1,
    )
    return scheduler


def prepare_train_data(
    uuids: List[str],
    data: Dict[str, Any],
    train_set: List[int],
    data_kwargs: Dict[str, Any],
    shuffle: bool = False,
    max_item: Optional[int] = None,
) -> Tuple[DynamicBatchDataset, DataLoader]:
    """
    Prepares training data and data loader for the training process.

    Parameters
    ----------
    uuids : List[str]
        List of unique identifiers for the data.
    data : Dict[str, Any]
        Dictionary containing the data.
    train_set : List[int]
        List of indices for the training set.
    data_kwargs : Dict[str, Any]
        Additional keyword arguments for data processing.
    shuffle : bool, optional
        Whether to shuffle the data, by default False.
    max_item : Optional[int], optional
        Maximum number of items to process, by default None.

    Returns
    -------
    Tuple[DynamicBatchDataset, DataLoader]
        The prepared training data and data loader.
    """
    # Create a test dataset with training data
    train_data = TestDataset(
        uuids=uuids[train_set], data=data, is_train=True, **data_kwargs
    )

    # Determine the maximum number of items based on data length
    max_item = 4 if (train_data.maxlen <= 512) else 2

    # Calculate the sizes for each training example
    train_sizes = [
        math.ceil(
            (max(len(data[uuid][0]) - train_data.maxlen + 2, 0) + train_data.stride)
            / train_data.stride
        )
        for uuid in train_data.uuids
    ]

    # Adjust training sizes based on the max_item parameter
    train_sizes = [
        size if max_item is None else min(size, max_item) for size in train_sizes
    ]

    # Create a dynamic batch dataset with the calculated sizes
    train_data = DynamicBatchDataset(
        train_data,
        batch_size=max(1, cfg.TRAIN_BATCH_SIZE // max(1, cfg.TRAIN_NUM_WORKERS)),
        sizes=train_sizes,
    )

    # Create a data loader for the training data
    train_loader = DataLoader(
        train_data,
        batch_size=max(1, cfg.TRAIN_NUM_WORKERS),
        num_workers=cfg.TRAIN_NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(
            collate_fn_list, pad_token_id=data_kwargs["pad_token_id"], max_item=max_item
        ),
    )

    return train_data, train_loader


def prepare_val_data(
    uuids: List[str],
    data: Dict[str, Any],
    val_set: List[int],
    data_kwargs: Dict[str, Any],
    max_item: int = 4,
) -> Tuple[DynamicBatchDataset, DataLoader]:
    """
    Prepare validation data for model evaluation.

    Parameters
    ----------
    uuids : List[str]
        List of unique identifiers for the data samples.
    data : Dict[str, Any]
        Dictionary containing the dataset.
    val_set : List[int]
        List of indices for validation set.
    data_kwargs : Dict[str, Any]
        Additional keyword arguments for data processing.
    max_item : int, optional
        Maximum number of items per batch, by default 4.

    Returns
    -------
    Tuple[DynamicBatchDataset, DataLoader]
        A tuple containing the validation dataset and data loader.
    """
    # Sort validation set by length of the text data (in descending order)
    val_set = sorted(val_set, key=lambda idx: -len(read_from_id(uuids[idx]).split()))

    # Initialize validation dataset
    val_data = TestDataset(
        uuids=uuids[val_set], data=data, is_train=False, **data_kwargs
    )

    # Adjust max_item based on the maximum sequence length in validation data
    max_item = 4 if (val_data.maxlen <= 512) else 2

    # Calculate sizes for dynamic batching
    val_sizes = [
        math.ceil(
            (max(len(data[uuid][0]) - val_data.maxlen + 2, 0) + val_data.stride)
            / val_data.stride
        )
        for uuid in val_data.uuids
    ]

    # Apply max_item constraint to sizes
    val_sizes = [
        size if max_item is None else min(size, max_item) for size in val_sizes
    ]

    # Initialize dynamic batch dataset
    val_data = DynamicBatchDataset(
        val_data,
        batch_size=max(1, cfg.VAL_BATCH_SIZE // max(1, cfg.VAL_NUM_WORKERS)),
        sizes=val_sizes,
    )

    # Initialize data loader for validation
    val_loader = DataLoader(
        val_data,
        batch_size=max(1, cfg.VAL_NUM_WORKERS),
        num_workers=cfg.VAL_NUM_WORKERS,
        collate_fn=partial(
            collate_fn_list, pad_token_id=data_kwargs["pad_token_id"], max_item=max_item
        ),
        shuffle=False,
    )

    return val_data, val_loader


def print_and_save(
    fold: int,
    oof: dict,
    metrics: dict,
    epoch: int,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    val_set: pd.DataFrame,
    train_set: pd.DataFrame,
    epochs_bar: object,
    do_save: bool,
    saver: object,
) -> None:
    """
    Logs and prints metrics, saves model and optimizer state if required.

    Parameters
    ----------
    fold : int
        Current fold number.
    oof : dict
        Dictionary to store out-of-fold data.
    metrics : dict
        Dictionary containing metrics values.
    epoch : int
        Current epoch number.
    net : torch.nn.Module
        The neural network model.
    optimizer : torch.optim.Optimizer
        Optimizer for the model.
    criterion : torch.nn.Module
        Loss function.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    val_set : pd.DataFrame
        Validation dataset.
    train_set : pd.DataFrame
        Training dataset.
    epochs_bar : object
        Progress bar for epochs.
    do_save : bool
        Flag to determine if the model should be saved.
    saver : object
        Object responsible for saving the model and logs.
    """

    # Update metrics dictionary
    metrics["epoch"] = epoch
    metrics["learning_rates"] = optimizer.param_groups[0]["lr"]

    # Update out-of-fold (oof) dictionary
    oof["val_set"] = val_set
    oof["train_set"] = train_set
    oof["fold"] = fold
    oof["criterion"] = criterion.__class__.__name__
    oof["model"] = net.__class__.__name__
    oof["scheduler"] = scheduler.__class__.__name__

    # Dictionary for formatting metrics
    metrics_format_dict = dict(
        loss="({loss:.6f}, {loss_val:.5f})",
        iov="(-1., {iov_val:.3f})",
        iov2="(-1., {iov_v2_val:.3f})",
        acc="({acc:.3f}, {acc_val:.3f})",
        f1="({f1:.3f}, {f1_val:.3f})",
        eff_acc="(-1, {acc_eff_val:.3f})",
        eff_f1="(-1, {f1_eff_val:.3f})",
    )

    # Format for printing metrics
    metrics_print_format = "[{epoch:02d}] loss: {loss} iov: {iov} iov2: {iov2} acc: {acc} f1: {f1} eff_acc: {eff_acc} eff_f1: {eff_f1}"

    # Format metrics values
    metrics_formated = {
        key: val.format(**metrics) for key, val in metrics_format_dict.items()
    }

    # Update progress bar with formatted metrics
    epochs_bar.set_postfix(**metrics_formated)

    # Print formatted metrics
    logger.info(metrics_print_format.format(epoch=epoch, **metrics_formated))

    # Save model and logs if do_save is True
    if do_save:
        saver.log(net, metrics, oof=oof)


def one_fold(
    *,
    uuids: List[str],
    data: Any,
    model_name: str,
    df: pd.DataFrame,
    fold: int,
    train_set: List[int],
    val_set: List[int],
    tokenizer: Any,
    model_config: Any,
    epochs: int = 20,
    save: bool = True,
    save_root: Union[str, Path] = None,
    checkpoint_paths: Optional[Dict[str, Union[str, Path]]] = None,
    use_stride_during_train: bool = False,
    use_position_embeddings: bool = True,
    save_each: Optional[int] = None,
    early_stop_epoch: Optional[int] = None,
    **data_kwargs,
) -> None:
    """
    Train a model for one fold of cross-validation.

    Parameters
    ----------
    uuids : List[str]
        List of unique identifiers for the data.
    data : Any
        The dataset to be used.
    model_name : str
        The name of the model to be trained.
    df : pd.DataFrame
        DataFrame containing the data.
    fold : int
        The fold number for cross-validation.
    train_set : List[int]
        Indices of the training set.
    val_set : List[int]
        Indices of the validation set.
    tokenizer : Any
        Tokenizer to be used with the model.
    model_config : Any
        Configuration for the model.
    epochs : int, optional
        Number of training epochs (default is 20).
    save : bool, optional
        Whether to save the model (default is True).
    save_root : Union[str, Path], optional
        Path to save the model (default is None).
    checkpoint_paths : Optional[Dict[str, str]], optional
        Paths to model checkpoints (default is None). If provided, used to
        initiate model weights.
    use_stride_during_train : bool, optional
        Whether to use stride during training (default is False). Striding allows the
        model to handle long text inputs, dividing them into smaller chunks uising a
        sliding window technique.
    use_position_embeddings : bool, optional
        Whether to use position embeddings (default is True).
    save_each : Optional[int], optional
        Save the model at each specified epoch (default is None).
    early_stop_epoch : Optional[int], optional
        Early stopping epoch (default is None).
    **data_kwargs : dict
        Additional keyword arguments for data preparation.

    Returns
    -------
    None
    """

    model_name_slug = slugify(model_name)

    save_root = Path(save_root) or cfg.MODEL_ROOT

    saver = AutoSave(
        root=save_root,
        name=f"fprize_{model_name_slug}_fold{fold}",
        metric=cfg.MAIN_METRIC_NAME,
    )

    checkpoint_path = (checkpoint_paths or {}).get(f"fold_{fold}")
    net = Model(
        model_name,
        checkpoint_path=checkpoint_path,
        pretrained=True,
        use_position_embeddings=use_position_embeddings,
        tokenizer=deepcopy(tokenizer),
        config=deepcopy(model_config),
    )

    check_if_model_has_position_embeddings(
        net, use_position_embeddings=use_position_embeddings
    )

    try:
        net = models_load_model(
            model=net, checkpoint_path=checkpoint_path, verbose=True
        )
    except Exception as e:
        logger.warn("Second Load ERRORRR:\n", str(e)[:500])

    if net.config.vocab_size < len(net.tokenizer):
        net.model.resize_token_embeddings(len(net.tokenizer))

    net.config.save_pretrained(save_root)
    net.tokenizer.save_pretrained(save_root)

    net = net.to(cfg.DEVICE)

    epochs_bar = tqdm(
        list(range(epochs if early_stop_epoch is None else early_stop_epoch)),
        leave=False,
    )

    if use_stride_during_train:
        _, train_loader = prepare_train_data(
            uuids=uuids,
            data=data,
            train_set=train_set,
            data_kwargs=data_kwargs,
            shuffle=False,
        )
    else:
        train_data = Dataset(
            uuids=uuids[train_set], data=data, is_train=True, **data_kwargs
        )
        train_loader = DataLoader(
            train_data,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            num_workers=cfg.TRAIN_NUM_WORKERS,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=partial(
                collate_fn_train, pad_token_id=data_kwargs["pad_token_id"]
            ),
        )

    _, val_loader = prepare_val_data(
        uuids=uuids, data=data, val_set=val_set, data_kwargs=data_kwargs
    )

    num_iters = len(train_loader) * epochs
    criterion = NERSegmentationLoss(num_iters=int(0.75 * num_iters))
    optimizer = fetch_optimizer(net)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=cfg.SCHEDULER_ETA_MIN, T_max=num_iters
    )
    schedule_each_step = True

    if cfg.USE_AMP and ("cuda" in str(cfg.DEVICE)):
        scaler = amp.GradScaler()
        warn("amp and fp16 are enabled !")
    else:
        scaler = None

    for epoch in epochs_bar:

        epochs_bar.set_description(f"--> [EPOCH {epoch:02d}]")
        net.train()

        oof, metrics = one_epoch(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            val_loader=val_loader,
            df=df,
            schedule_each_step=schedule_each_step,
            save_each=save_each,
            # For print and save
            fold=fold,
            epoch=epoch,
            val_set=val_set,
            train_set=train_set,
            epochs_bar=epochs_bar,
            do_save=save,
            saver=saver,
        )

        if use_stride_during_train:
            # We need to redefine train data with custom shuffle because data loader shuffling is disabled cause of DynamicBatchDataset
            _, train_loader = prepare_train_data(
                uuids=uuids,
                data=data,
                train_set=train_set[np.random.permutation(len(train_set))],
                data_kwargs=data_kwargs,
                shuffle=False,
            )


def train(
    *,
    uuids: Union[Dict[str, int], List[str]],
    data: Dict[str, Any],
    model_name: str,
    df: pd.DataFrame,
    tokenizer: Any,
    model_config: dict,
    epochs: int = 20,
    save: bool = True,
    n_splits: int = 5,
    seed: Optional[int] = None,
    save_root: Union[str, None] = None,
    suffix: str = "",
    folds: Optional[List[int]] = None,
    checkpoint_paths: Optional[Dict[str, Union[str, Path]]] = None,
    use_stride_during_train: bool = False,
    use_position_embeddings: bool = True,
    save_each: Optional[int] = None,
    **data_kwargs,
) -> None:
    """
    Train the model with k-fold cross-validation.

    Parameters
    ----------
    uuids : Union[Dict[str, int],List[str]]
        Unique identifiers for the data samples.
    data : Dict[str: Any]
        The data to be used for training.
    model_name : str
        Name of the model.
    df : pd.DataFrame
        DataFrame containing the data.
    tokenizer : Any
        Tokenizer to be used.
    model_config : dict
        Configuration of the model.
    epochs : int, optional
        Number of epochs for training, by default 20.
    save : bool, optional
        Whether to save the model, by default True.
    n_splits : int, optional
        Number of splits for k-fold, by default 5.
    seed : int, optional
        Seed for reproducibility, by default None.
    save_root : Union[str, None], optional
        Root directory to save the model, by default None.
    suffix : str, optional
        Suffix for the save directory, by default "".
    folds : List[int], optional
        Specific folds to run, by default None.
    checkpoint_paths : Optional[Dict[str, str]], optional
        Paths to model checkpoints (default is None). If provided, used to
        initiate model weights.
    use_stride_during_train : bool, optional
        Whether to use stride during training (default is False). Striding allows the
        model to handle long text inputs, dividing them into smaller chunks uising a
        sliding window technique.
    use_position_embeddings : bool, optional
        Whether to use position embeddings (default is True).
    save_each : int, optional
        Interval to save the model, by default None.
    **data_kwargs : dict
        Additional data keyword arguments.
    """
    # Collect garbage and empty CUDA cache
    gc.collect()
    torch.cuda.empty_cache()

    seed = cfg.SEED if seed is None else seed

    # Create save directory
    model_name_slug = slugify(model_name)
    save_root = save_root or cfg.MODEL_ROOT / f"{model_name_slug}{suffix}"
    save_root.mkdir(exist_ok=True, parents=True)

    # Set random seed for reproducibility
    seed_everything(seed)

    # Prepare fold splits
    if isinstance(uuids, dict):
        fold_bar = []
        n_splits = max(uuids.values()) + 1

        for fold in range(n_splits):
            train_set = np.array(
                [i for i, fold_i in enumerate(uuids.values()) if fold_i != fold],
                dtype=np.int64,
            )

            val_set = np.array(
                [i for i, fold_i in enumerate(uuids.values()) if fold_i == fold],
                dtype=np.int64,
            )

            fold_bar.append((train_set, val_set))

        uuids = np.array(list(uuids.keys()))
    else:
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        fold_bar = list(kf.split(np.arange(len(uuids))))

    if folds:
        fold_bar = tqdm([(fold, fold_bar[fold]) for fold in folds])
    else:
        fold_bar = tqdm(enumerate(fold_bar), total=n_splits)

    # Train on each fold
    for fold, (train_set, val_set) in fold_bar:
        logger.info(f"\n############################### [FOLD {fold}  SEED {seed}]")
        fold_bar.set_description(f"[FOLD {fold}  SEED {seed}]")

        one_fold(
            uuids=uuids,
            data=data,
            model_name=model_name,
            df=df,
            fold=fold,
            train_set=train_set,
            val_set=val_set,
            epochs=epochs,
            save=save,
            save_root=save_root,
            checkpoint_paths=checkpoint_paths,
            use_stride_during_train=use_stride_during_train,
            use_position_embeddings=use_position_embeddings,
            save_each=save_each,
            tokenizer=tokenizer,
            model_config=model_config,
            **data_kwargs,
        )

        # Collect garbage and empty CUDA cache after each fold
        gc.collect()
        torch.cuda.empty_cache()
