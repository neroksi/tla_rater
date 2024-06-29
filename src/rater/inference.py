import gc
import math
from functools import partial
from pathlib import Path
from typing import Dict
from warnings import warn

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from . import configs as cfg
from .dataset import (
    DynamicBatchDataset,
    RunningMax,
    TestDataset,
    collate_fn_list,
    get_append_bar_chars,
)
from .models import Model, check_if_model_has_position_embeddings
from .post_processing import make_sub_from_res
from .script_utils import mp_gen_data
from .utils import copy_param_to_configs


def load_net(checkpoint_path, param: dict):
    config = param["config"]
    use_position_embeddings = param["use_position_embeddings"]

    if use_position_embeddings:
        config.position_biased_input = True
        config.relative_attention = True

    net = Model(
        config=config,
        pretrained=False,
        use_position_embeddings=use_position_embeddings,
        tokenizer=param["tokenizer"],
    )

    net = net.to(cfg.DEVICE)

    check_if_model_has_position_embeddings(
        net, use_position_embeddings=use_position_embeddings
    )

    if checkpoint_path is not None:
        net.load_state_dict(
            torch.load(checkpoint_path, map_location=cfg.DEVICE),
            strict=param.get("strict", True),
        )
    net = net.eval()
    return net


def get_params(model_name: str, **kwargs):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # models = sorted(Path(model_name).glob("*.pth"))

    params = {
        "model_name": model_name,
        "config": config,
        "tokenizer": tokenizer,
    }

    params.update(kwargs)

    return params


def get_max_pads(attention_mask):
    bools = (attention_mask[:, 1:] == 0) & (attention_mask[:, :-1] != 0)
    max_pads = bools.to(torch.uint8).argmax(1)

    if (max_pads == 0).any():  # There at least one full batch, no padding
        max_pads = attention_mask.shape[1]
    else:
        max_pads = 1 + max_pads.max().item()

    return max_pads


@torch.no_grad()
def _predict(
    nets,
    input_ids,
    attention_mask,
    word_ids,
    agg="mean",
    return_output=False,
    dynamic_padding=True,
    apply_softmax=False,
):

    if return_output:
        assert isinstance(nets, torch.nn.Module)
        nets = [nets]

    if len(nets) > 1:
        assert apply_softmax

    pred = pred_seg = pred_eff = None
    for net in nets:
        o_all = net(
            input_ids=input_ids, attention_mask=attention_mask, word_ids=word_ids
        )
        o, o_seg, o_eff = o_all.out_ner, o_all.out_seg, o_all.out_eff

        if apply_softmax:
            o, o_seg = o.softmax(dim=-1), o_seg.softmax(dim=-1)
            o_eff = o_eff.softmax(dim=-1)

        if agg == "max":
            pred = o if pred is None else torch.max(o, pred)
            pred_seg = o_seg if pred_seg is None else torch.max(o, pred_seg)
            pred_eff = o_eff if pred_eff is None else torch.max(o, pred_eff)
        elif agg == "mean":
            pred = o if pred is None else pred.add_(o)
            pred_seg = o_seg if pred_seg is None else pred_seg.add_(o_seg)
            pred_eff = o_eff if pred_eff is None else pred_eff.add_(o_eff)
        else:
            raise ValueError(f"Unknow value `{agg}` for `agg`")

    if agg == "mean":
        pred /= len(nets)
        pred_seg /= len(nets)
        pred_eff /= len(nets)

    return (
        (pred, pred_seg, pred_eff, o_all)
        if return_output
        else (pred, pred_seg, pred_eff)
    )


@torch.no_grad()
def predict_eval(net, test_data, bar=False, ret_out=True, dynamic_padding=True):
    rm = RunningMax()
    rm_seg = RunningMax()
    rm_eff = RunningMax()
    rm_target = RunningMax()
    rm_eff_target = RunningMax()

    if ret_out:
        out_list = []

    target_list = []

    test_data = tqdm(test_data, desc="PREDICT") if bar else test_data

    apply_softmax = not ret_out

    for i_inp, inp in enumerate(test_data):
        assert len(inp) in [5, 7], f"``{len(inp)}`` is an unknown number of elements!"

        batch_ids, word_ids, word_perc_pos, input_ids, attention_mask = inp[:5]

        target, eff_target = inp[5:] if len(inp) == 7 else (None, None)

        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
            attention_mask = torch.from_numpy(attention_mask)
            word_perc_pos = torch.from_numpy(word_perc_pos)

        input_ids = input_ids.to(cfg.DEVICE)  # .long()
        attention_mask = attention_mask.to(cfg.DEVICE)  # .long()
        word_perc_pos = word_perc_pos.to(cfg.DEVICE)

        preds, preds_seg, preds_eff, out = _predict(
            net,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_perc_pos,
            return_output=True,
            dynamic_padding=dynamic_padding,
            apply_softmax=apply_softmax,
        )

        if not apply_softmax:
            preds, preds_seg = preds.softmax(dim=-1), preds_seg.softmax(dim=-1)
            preds_eff = preds_eff.softmax(dim=-1)

        preds, preds_seg, preds_eff = (
            preds.squeeze(1).cpu().float().numpy(),
            preds_seg.squeeze(1).cpu().float().numpy(),
            preds_eff.squeeze(1).cpu().float().numpy(),
        )

        if dynamic_padding and (preds.shape[1] < word_ids.shape[1]):
            word_ids = word_ids[:, : preds.shape[1]]
            if target is not None:
                target = target[:, : preds.shape[1]]
                eff_target = eff_target[:, : preds.shape[1]]

        if ret_out:
            out_list.append(out)

        if target is not None:
            target_list.append(target)

        bools = word_ids >= 0

        batch_ids = batch_ids[:, None].repeat(word_ids.shape[1], axis=1)[bools]
        word_ids = word_ids[bools]

        reduce = i_inp % 200 == 0
        red_ids = np.c_[batch_ids, word_ids]
        rm.update(ids=red_ids, vals=preds[bools], reduce=reduce)

        rm_seg.update(ids=red_ids, vals=preds_seg[bools], reduce=reduce)
        rm_eff.update(ids=red_ids, vals=preds_eff[bools], reduce=reduce)

        if target is not None:
            rm_target.update(
                ids=red_ids,
                vals=target[bools][:, None].astype(np.int8, copy=False),
                reduce=reduce,
            )

            rm_eff_target.update(
                ids=red_ids,
                vals=eff_target[bools][:, None].astype(np.int8, copy=False),
                reduce=reduce,
            )

    if len(rm._vals):
        rm.reduce()
        rm_seg.reduce()
        rm_eff.reduce()

        if target is not None:
            rm_target.reduce()
            rm_eff_target.reduce()

    preds = rm.normlaize().vals
    preds_seg = rm_seg.normlaize().vals
    preds_eff = rm_eff.normlaize().vals

    target_v2 = rm_target.vals if target is not None else None
    eff_target_v2 = rm_eff_target.vals if eff_target is not None else None

    if dynamic_padding:
        out = out_list if ret_out else None
        target = target_list if target is not None else None
    else:
        out = tuple(map(torch.cat, zip(*out_list))) if ret_out else None
        target = (
            torch.from_numpy(np.concatenate(target_list)).to(cfg.DEVICE)
            if target is not None
            else None
        )

    res = {
        "out": out,
        "target": target,
        "preds": preds,
        "preds_seg": preds_seg,
        "preds_eff": preds_eff,
        "target_v2": target_v2,
        "eff_target_v2": eff_target_v2,
    }
    return res


def predict_from_param(
    uuids,
    param,
    data=None,
    texts: Dict[str, str] = None,
    oof=False,
    make_sub=True,
    dynamic_padding=True,
    model_bar=False,
    fp16=False,
):
    copy_param_to_configs(param)

    assert not oof or not make_sub

    warn("You should sort your UUIDs for faster prediction")

    tokenizer = param["tokenizer"]
    pad_token_id = (tokenizer.pad_token_id,)
    special_token_ids = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if data is None:
        append_bar = get_append_bar_chars(tokenizer=tokenizer)

        data = mp_gen_data(
            uuids,
            tokenizer=tokenizer,
            df=None,
            texts=texts,
            append_bar=append_bar,
            root=param.get("root"),
        )

    test_data = TestDataset(
        uuids=uuids,
        data=data,
        pad_token_id=pad_token_id,
        tokenizer=tokenizer,
        root=param.get("root"),
        stride=param["stride"],
        special_token_ids=special_token_ids,
    )

    sizes = [
        math.ceil(
            (max(len(data[uuid][0]) - test_data.maxlen + 2, 0) + test_data.stride)
            / test_data.stride
        )
        for uuid in uuids
    ]

    test_data = DynamicBatchDataset(
        test_data,
        batch_size=max(1, param["batch_size"] // max(1, param["num_workers"])),
        sizes=sizes,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=max(1, param["num_workers"]),
        num_workers=param["num_workers"],
        shuffle=False,
        collate_fn=partial(collate_fn_list, pad_token_id=pad_token_id),
    )

    preds = preds_seg = preds_eff = None

    for model_path in tqdm(param["model_paths"], desc="Inference::Models"):
        model = load_net(model_path, param)

        if fp16:
            # model = model.half()
            model = model.bfloat16()

        res = predict_eval(
            model,
            test_loader,
            bar=model_bar,
            ret_out=False,
            dynamic_padding=dynamic_padding,
        )

        if preds is None:
            preds = res["preds"]
            preds_seg = res["preds_seg"]
            preds_eff = res["preds_eff"]
        else:
            preds += res["preds"]
            preds_seg += res["preds_seg"]
            preds_eff += res["preds_eff"]

        del model, res

        gc.collect()
        torch.cuda.empty_cache()

    preds /= len(param["model_paths"])
    preds_seg /= len(param["model_paths"])
    preds_eff /= len(param["model_paths"])

    if make_sub:

        sub = make_sub_from_res(
            uuids=uuids,
            res=preds,
            res_seg=preds_seg,
        )

        return preds, preds_seg, preds_eff, sub

    return preds, preds_seg, preds_eff
