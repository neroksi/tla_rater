import re
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from . import configs as cfg
from .configs import logger

mTaskOutputType = namedtuple("mTaskOutputType", "out_ner out_seg out_eff")


def check_if_model_has_position_embeddings(
    model: torch.nn.Module, use_position_embeddings: Optional[bool] = None
) -> bool:
    """
    Check if the model has position embeddings and ensure compatibility with `use_position_embeddings`.

    Parameters
    ----------
    model : torch.nn.Module
        The model to check for position embeddings.
    use_position_embeddings : Optional[bool], optional
        Flag to enforce the presence or absence of position embeddings in the model. Default is None.

    Returns
    -------
    bool
        True if the model has position embeddings, False otherwise.

    Raises
    ------
    ValueError
        If `use_position_embeddings` is True and the model does not have position embeddings,
        or if `use_position_embeddings` is False and the model has position embeddings.
    """
    has_position_embeddings = False
    for name, param in model.named_parameters():
        if "position_embeddings" in name.lower():
            has_position_embeddings = True
            break

    if use_position_embeddings is not None:
        if use_position_embeddings and not has_position_embeddings:
            raise ValueError(
                "Position embeddings is required but the current model has NOT."
            )

        if not use_position_embeddings and has_position_embeddings:
            raise ValueError(
                "Position embedding is turned off but the current model has one."
            )

    if has_position_embeddings:
        logger.warn("The current model has <position_embeddings> enabled.")
    else:
        logger.warn("The current model has NOT <position_embeddings> enabled.")

    return has_position_embeddings


def load_model_weights(
    model_class: type = None,
    model: torch.nn.Module = None,
    checkpoint_path: str = None,
    verbose: bool = False,
    remove: str = None,
    match: str = None,
    **kwargs,
) -> torch.nn.Module:
    """
    Load a model with specified parameters and optional checkpoint.

    Parameters
    ----------
    model_class : type, optional
        The class of the model to be instantiated if model is not provided.
    model : torch.nn.Module, optional
        The model instance to load the weights into.
    checkpoint_path : str, optional
        The path to the checkpoint file.
    verbose : bool, optional
        If True, prints additional information.
    remove : str, optional
        Regex pattern to remove from state dict keys.
    match : str, optional
        Regex pattern to match keys in the state dict.
    **kwargs
        Additional arguments for the model class if model is None.

    Returns
    -------
    torch.nn.Module
        The loaded model in evaluation mode.
    """
    DEVICE = torch.device("cpu")

    if model is None:
        assert model_class is not None
        model = model_class(**kwargs)

    model = model.to(DEVICE)

    if checkpoint_path is not None:
        weights_dict = torch.load(checkpoint_path, map_location=DEVICE)

        for key in list(weights_dict):
            if match and not re.search(match, key):
                weights_dict.pop(key)
            elif remove:
                key2 = re.sub(remove, "", key)
                weights_dict[key2] = weights_dict.pop(key)

        try:
            model.load_state_dict(weights_dict, strict=True)
        except Exception as e:
            logger.warn(
                f"Model loading in strict mode failed, will try in non-strict mode\n{str(e)[:500]}"
            )
            model.load_state_dict(weights_dict, strict=False)

        if verbose:
            logger.info(f"Weights loaded from: '{checkpoint_path}'")

    model = model.eval()
    return model


def get_model(
    model_name: Optional[str] = None,
    num_targets: Optional[int] = None,
    config: Optional[PretrainedConfig] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    pretrained: bool = False,
    use_position_embeddings: bool = True,
) -> Tuple[PretrainedConfig, Optional[AutoTokenizer], AutoModelForTokenClassification]:
    """
    Get a model for token classification.

    Parameters
    ----------
    model_name : Optional[str], default=None
        Name of the pretrained model.
    num_targets : Optional[int], default=None
        Number of target classes.
    config : Optional[AutoConfig], default=None
        Model configuration.
    tokenizer : Optional[AutoTokenizer], default=None
        Tokenizer for the model.
    pretrained : bool, default=False
        Whether to load a pretrained model.
    use_position_embeddings : bool, default=True
        Whether to use position embeddings.

    Returns
    -------
    Tuple[AutoConfig, Optional[AutoTokenizer], AutoModelForTokenClassification]
        The configuration, tokenizer, and model.
    """
    num_targets = cfg.NUM_TARGETS if num_targets is None else num_targets

    model_instance = AutoModelForTokenClassification

    if use_position_embeddings and config is None:
        assert model_name

        config = AutoConfig.from_pretrained(model_name)

    if use_position_embeddings:
        config.position_biased_input = True
        config.relative_attention = True
    elif config:
        config.position_biased_input = False

    if not pretrained:
        assert config is not None
        model = model_instance.from_config(config)
    else:
        assert model_name is not None

        config = AutoConfig.from_pretrained(model_name) if config is None else config
        tokenizer = (
            AutoTokenizer.from_pretrained(model_name, config=config)
            if tokenizer is None
            else tokenizer
        )
        model = model_instance.from_pretrained(model_name, config=config)

    if hasattr(model, "classifier"):
        model.classifier = nn.Linear(model.classifier.in_features, num_targets)

    return config, tokenizer, model


class MultiSampleDropout(nn.Module):
    """
    Implements multi-sample dropout with varying dropout probabilities.

    This is a technique that helps accelerate model convergence by outputing several
    predictions (instead of one) for the given target instead. If does so by applying
    different dropout mask for each prediction.

    Parameters
    ----------
    n_drops : Optional[int]
        Number of dropout samples.
    p_drops : Optional[List[float]]
        List of dropout probabilities.
    """

    def __init__(
        self, n_drops: Optional[int] = None, p_drops: Optional[List[float]] = None
    ) -> None:

        super().__init__()

        self.q_0 = 0.10
        self.q_1 = 0.50 - self.q_0

        self.n_drops = n_drops or cfg.N_DROPS
        self.p_drops = (p_drops or cfg.P_DROPS) or self.gen_dropout_probas()

        self.drop_modules = nn.ModuleList(
            [nn.Dropout(p_drop) for p_drop in self.p_drops]
        )

    def gen_dropout_probas(self) -> List[float]:
        """
        Generate dropout probabilities.

        Returns
        -------
        List[float]
            List of generated dropout probabilities.
        """
        assert self.n_drops >= 0

        if self.n_drops == 0:
            return []
        elif self.n_drops == 1:
            return [self.q_0]
        else:
            return [
                self.q_0 + self.q_1 * n / (self.n_drops - 1)
                for n in range(self.n_drops)
            ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiSampleDropout module.

        Parameters
        ----------
        x : torch.Tensor, BxTxD
            Input tensor.

        Returns
        -------
        torch.Tensor, BxMxTxD
            Output tensor with dropout applied.
        """
        if not self.training or not self.n_drops:
            return x[:, None]

        res = []
        for drop_module in self.drop_modules:
            res.append(drop_module(x))
        res = torch.stack(res, dim=1)
        return res


class BaseModel(nn.Module):
    """
    Base Model for token classification tasks.

    Parameters
    ----------
    model_name : Optional[str]
        Name of the model to be used.
    num_targets : Optional[int]
        Number of target classes.
    config : Optional[dict]
        Configuration dictionary for the model.
    tokenizer : Optional[PreTrainedTokenizer]
        Tokenizer to be used with the model.
    checkpoint_path : Optional[str]
        Path to the checkpoint file for loading model weights. If provided, will be used
        to initiate model weights.
    pretrained : bool, optional, default=False
        Whether to use pretrained weights.
    use_position_embeddings : bool, optional, default=True
        Whether to use position embeddings.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        num_targets: Optional[int] = None,
        config: Optional[PretrainedConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        checkpoint_path: Optional[str] = None,
        pretrained: bool = False,
        use_position_embeddings: bool = True,
    ):
        super().__init__()
        self.model_name = cfg.MODEL_NAME if model_name is None else model_name
        self.num_targets = cfg.NUM_TARGETS if num_targets is None else num_targets
        self.use_position_embeddings = use_position_embeddings

        config, tokenizer, model = get_model(
            model_name=model_name,
            config=config,
            tokenizer=tokenizer,
            num_targets=1,
            pretrained=pretrained,
            use_position_embeddings=use_position_embeddings,
        )

        self.in_features = model.classifier.in_features
        model.classifier = nn.Identity()

        self.config = config
        self.tokenizer = tokenizer
        self.model = model

        if checkpoint_path is not None:
            self.model = load_model_weights(
                model=self.model,
                checkpoint_path=checkpoint_path,
                verbose=True,
                remove=r"^model\.",
                match=r"^model\.",
            )

        # if config.vocab_size < len(tokenizer):
        #     self.model.resize_token_embeddings(len(tokenizer))

        self.fc = nn.Linear(self.in_features, self.num_targets)


class Model(BaseModel):
    """
    A model class for handling different tasks.


    Parameters
    ----------
    model_name : Optional[str], default=None
        Name of the model.
    num_targets : Optional[int], default=None
        Number of target variables.
    config : Optional[dict], default=None
        Configuration dictionary.
    tokenizer : Optional[Any], default=None
        Tokenizer to be used.
    checkpoint_path : Optional[str], default=None
        Path to the checkpoint file.
    pretrained : bool, default=False
        Flag to indicate if a pretrained model should be used.
    use_position_embeddings : bool, default=True
        Flag to indicate if position embeddings should be used.


    Attributes
    ----------
    _different_lr_s : list
        List of regular expressions to match the parameters that requires a different
        learning the global one. This usually the case for model's final linear heads.
    ms_dropout : MultiSampleDropout
        Instance of the MultiSampleDropout class.
    fc : torch.nn.Linear, dim ==> *x15
        Linear head for essay-type predicition. This is the most important layer since it's
        responsible for essay-type classification as NER task. Final output size is 15,
        ie 7 classes for `Beginning` targets, 7 other targets for `In` targets and 1 class
        for `Out`target. Indeed, we have adopted the IOB format (please see
        https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))
    fc_seg : torch.nn.Linear, dim ==> *x3
        Linear head for segmentation. The final output has a size of 3:
        0 ==> Out of any Entity
        1 ==> Beginning of an Entity (no matter which)
        2 ==> Inside of an Entity (no matter which)

        In fact, we also have this dedicated sequence segmentation head for a better Entity
        boudaries detection. Combined with the boundaries derived from the previous head,
        entity boundaries are better detected during inference.
    fc_eff : torch.nn.Linear, dim ==> *x2
        Linear head for effectiveness. This layer has been added to output the effectiveness
        of each span. During inference, a span's effectiveness score is computed by averaging
        scores from this layer. Hence all the three tasks of the competition are handled as token
        classification tasks and a small post-processing step is added during inference to put
        everything into their expected output format.
    """

    _different_lr_s: List[str] = []

    def __init__(
        self,
        model_name: Optional[str] = None,
        num_targets: Optional[int] = None,
        config: Optional[PretrainedConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        checkpoint_path: Optional[str] = None,
        pretrained: bool = False,
        use_position_embeddings: bool = True,
    ):

        super().__init__(
            model_name=model_name,
            num_targets=num_targets,
            config=config,
            tokenizer=tokenizer,
            checkpoint_path=checkpoint_path,
            pretrained=pretrained,
            use_position_embeddings=use_position_embeddings,
        )

        self.ms_dropout = MultiSampleDropout()

        self.fc_seg = nn.Linear(self.in_features, 3)
        self.fc_eff = nn.Linear(self.in_features, len(cfg.EFF2ID))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids: Optional[torch.Tensor],
    ) -> mTaskOutputType:
        """
        Forward pass of the model.

        Parameters
        ----------
        input_ids : torch.Tensor, BxT
            Tensor containing the input IDs.
        attention_mask : torch.Tensor, BxT
            Tensor containing the attention mask.
        word_ids : Optional[torch.Tensor], BxT
            Tensor containing the word IDs, required if position embeddings are used.
            word_ids could be viewed as proxies for transformers' position_ids. They're meant
            to work even for sequences longer than the model's `max_token` since the are normalized
            into the range (0, 1) before being multiplied by `max_token`.

        Returns
        -------
        mTaskOutputType
            Output of the model containing named entity recognition,  segmentation,  and effectiveness scores.
        """
        if self.use_position_embeddings:
            assert word_ids is not None
            position_ids = word_ids
        else:
            position_ids = None

        x = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[
            "logits"
        ]  # ==> BxTxD

        x = self.ms_dropout(x)  # ==> BxMxTxD, where M = number of multi-sample dropout

        out_seg = self.fc_seg(x)  # => BxMxTx3
        out_ner = self.fc(x)  # => BxMxTx15
        out_eff = self.fc_eff(x)  # => BxMxTx2

        out = mTaskOutputType(
            out_ner=out_ner,
            out_seg=out_seg,
            out_eff=out_eff,
        )

        return out


class NERSegmentationLoss(nn.Module):
    """
    A class to compute combined Named Entity Recognition (NER), Segmentation, and Effectiveness losses.

    Parameters
    ----------
    num_iters : int
        Number of total training iterations, useful when dynamic weighting is enabled.
    """

    def __init__(self, num_iters: int) -> None:
        super().__init__()

        self.num_iters = num_iters
        self.num_calls = 0

        self.ner_weights = torch.tensor(
            cfg.CLASS_WEIGHTS, device=cfg.DEVICE, dtype=torch.float32
        )
        self.seg_weights = torch.tensor(
            cfg.SEG_CLASS_WEIGHTS, device=cfg.DEVICE, dtype=torch.float32
        )

        # self.eff_weights = torch.tensor(
        #     [0.2, 0.8], device=cfg.DEVICE, dtype=torch.float32
        # )

    def coef(self, k: int) -> float:
        """
        Compute the coefficient for dynamic weighting.

        Dynamic weighting implies assigning dymically decreasing weights to
        classes until all of them reach same weighit of 1.0 near the eand of
        training. This ensures that over-represented classes will get smaller
        importance at the beginning of the training but these weights will be adjusted
        over time to let the model find the **right equilibrum** and avoid bias and
        calibration issues in the final model outputs.

        Parameters
        ----------
        k : int
            Current iteration count.

        Returns
        -------
        float
            Cosine based synamic weiht coefficient.
        """
        return max(0, np.cos(np.pi * k / (2 * self.num_iters)))

    def get_losses(
        self, num_calls: Optional[int] = None
    ) -> Tuple[nn.CrossEntropyLoss, nn.CrossEntropyLoss, nn.CrossEntropyLoss]:
        """
        Get loss functions for NER, segmentation, and effectiveness.

        Some of the losses have dynamic weighting enabled, please see above
        for more details on dynamic weight computation.

        Parameters
        ----------
        num_calls : int, optional
            Number of calls to this function. Defaults to self.num_calls.

        Returns
        -------
        Tuple[nn.CrossEntropyLoss, nn.CrossEntropyLoss, nn.CrossEntropyLoss]
            NER, segmentation, and effectiveness loss functions.
        """
        num_calls = self.num_calls if num_calls is None else num_calls
        q = self.coef(num_calls)
        ner_loss = nn.CrossEntropyLoss(weight=(1 - q) + q * self.ner_weights)
        seg_loss = nn.CrossEntropyLoss(weight=(1 - q) + q * self.seg_weights)
        eff_loss = nn.CrossEntropyLoss(
            # weight=(1 - q) + q * self.eff_weights
        )
        return ner_loss, seg_loss, eff_loss

    @staticmethod
    def generate_seg_target(target: torch.Tensor) -> torch.Tensor:
        """
        Generate segmentation target labels from NER target.

        Parameters
        ----------
        target : torch.Tensor, BxT, in [0, 15[
            Original target labels.

        Returns
        -------
        torch.Tensor, BxT, in [0, 3[
            Class-blind segmentation target labels.
        """
        class_blind_labels = (
            1 * ((1 <= target) & (target < 8))  # In
            + 2 * ((8 <= target) & (target < 15))  # Beginning
            + 1 * ((15 <= target) & (target < 22))  # In
        )
        if isinstance(class_blind_labels, torch.Tensor):
            class_blind_labels = class_blind_labels.long()

        bools = target <= 0
        class_blind_labels[bools] = target[bools]
        return class_blind_labels

    def forward(
        self,
        out_ner: torch.Tensor,
        out_seg: torch.Tensor,
        out_eff: torch.Tensor,
        target: torch.Tensor,
        eff_target: torch.Tensor,
        num_calls: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute the combined loss.

        Parameters
        ----------
        out_ner : torch.Tensor, BxMxTx15
            Output logits for NER. Please see model definition for why final dim is 15.
        out_seg : torch.Tensor, BxMxTx3
            Output logits for segmentation. Please see model definition for why final dim is 3.
        out_eff : torch.Tensor, BxMxTx2
            Output logits for effectiveness. Please see model definition for why final dim is 2.
        target : torch.Tensor, BxT, in [0, 15[
            Target labels for NER.
        eff_target : torch.Tensor, BxT, in [0, 2[
            Target labels for effectiveness.
        num_calls : int, optional
            Number of calls to this function. Defaults to None.

        Returns
        -------
        torch.Tensor
            Combined loss.
        """
        ner_loss, seg_loss, eff_loss = self.get_losses(num_calls=num_calls)

        seg_labels = self.generate_seg_target(target)  # BxT, values are in [0, 3[

        # Adding multi-sample-dropout dimension
        seg_labels = torch.stack([seg_labels] * out_seg.size(1), dim=1)  # => BxMxT
        target = torch.stack([target] * out_ner.size(1), dim=1)  # => BxMxT
        eff_target = torch.stack([eff_target] * out_eff.size(1), dim=1)  # => BxMxT

        # Filter out and compute losses
        bools = target >= 0
        out_ner, out_seg, target, seg_labels = (
            out_ner[bools],
            out_seg[bools],
            target[bools],
            seg_labels[bools],
        )

        # let N == product(B, M, T) minus Number of Excluded Tokens (ie target < 0)

        l_ner = ner_loss(out_ner, target)  # => Nx15, N
        l_seg = seg_loss(out_seg, seg_labels)  # => Nx3, N

        # let R == product(B, M, T) - number of Excluded Tokens (ie eff_target < 0)
        eff_bools = eff_target >= 0
        l_eff = eff_loss(out_eff[eff_bools], eff_target[eff_bools])  # => Rx2, R

        l_all = cfg.ALPHA_NER * l_ner + cfg.ALPHA_SEG * l_seg + cfg.ALPHA_EFF * l_eff

        self.num_calls += 1  # increment total number of calls

        return l_all
