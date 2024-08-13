import bisect
import re
import unicodedata
from itertools import chain as it_chain
from pathlib import Path
from string import punctuation
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset as TorchDataset

from . import configs as cfg

puncs = re.escape(punctuation)
_split_regex = r"(\s+)"


def read_from_id(text_id: str, root: Optional[Union[str, Path]] = None) -> str:
    """
    Reads and normalizes text from a file with the given ID.

    Parameters
    ----------
    text_id : str
        The identifier for the text file.
    root : Optional[Union[str, Path]], optional
        The root directory where the text file is located. If None, uses the default
        training root directory defined in cfg.TRAIN_ROOT.

    Returns
    -------
    str
        The normalized text content of the file.
    """
    # Set root to the provided path or default to cfg.TRAIN_ROOT if not provided
    root = Path(root or cfg.TRAIN_ROOT)

    # Read the text from the file, normalize it, and return
    return unicodedata.normalize(
        "NFKD", (root / text_id).with_suffix(".txt").read_text(encoding="utf-8")
    )


def read_train_df(train_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load a training dataframe from a CSV file.

    Parameters
    ----------
    train_csv_path : Optional[str]
        The file path to the training CSV. If None, the path is taken from the configuration.

    Returns
    -------
    pd.DataFrame
        The training dataframe with necessary data type conversions and mappings applied.
    """
    # Use default path from config if no path provided
    train_csv_path = train_csv_path or cfg.TRAIN_CSV_PATH

    # Read the CSV file
    df = pd.read_csv(train_csv_path, encoding="utf-8")

    # Convert 'discourse_start' and 'discourse_end' to integer
    df["discourse_start"] = df["discourse_start"].astype(int)
    df["discourse_end"] = df["discourse_end"].astype(int)

    # Process 'predictionstring' to numpy array, handle missing values
    df["predictionstring"] = (
        df["predictionstring"]
        .fillna("")  # Fill missing values with empty strings
        .astype(str)  # Ensure all entries are strings
        .apply(
            lambda x: np.fromstring(x, sep=" ", dtype=np.int16)
        )  # Convert to numpy array
    )

    # Map 'discourse_type' to an integer ID using a configuration mapping
    df["discourse_type_id"] = df["discourse_type"].map(cfg.DISCOURSE2ID)

    return df


def add_training_fold(
    df: pd.DataFrame,
    nfolds: int = 5,
    group_col: str = "essay_id",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Adds a 'fold' column to the DataFrame based on stratified group k-folds.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing at least the column specified by `group_col`.
    nfolds : int, default=5
        The number of folds to create.
    group_col : str, default="essay_id"
        The name of the column to use for grouping data.
    seed : Optional[int], default=None
        The seed for random number generator. Uses cfg.SEED if None.

    Returns
    -------
    pd.DataFrame
        The DataFrame with an added 'fold' column indicating the fold assignment.

    Notes
    -----
    - The 'fold' column is initialized with -1 to indicate unassigned folds.
    - GroupKFold from sklearn is used to ensure groups specified by `group_col` are not split across folds.
    """
    seed = seed or cfg.SEED
    np.random.seed(seed)

    df["fold"] = -1  # Initialize fold column with -1

    gkf = GroupKFold(n_splits=nfolds)

    # Assign folds using GroupKFold
    for fold, (_, val_set) in enumerate(
        gkf.split(np.arange(len(df)), groups=df[group_col])
    ):
        df.loc[df.index[val_set], "fold"] = fold

    return df


def split(s: str) -> List[str]:
    """
    Splits the input string based on a pre-defined regular expression.

    Parameters
    ----------
    s : str
        The string to be split.

    Returns
    -------
    List[str]
        A list of substrings after splitting the original string based on regex.

    Notes
    -----
    - The function uses a global regex pattern `_split_regex` for splitting.
    - Adjacent segments not matching the regex are concatenated before appending.
    """
    # Split the string using the global regex pattern
    res = re.split(_split_regex, s)
    new_res = []
    t = ""

    # Process each segment from the split results
    for r in res:
        # Check if current segment matches the regex and `t` is not empty
        if re.match(_split_regex, r) and t:
            new_res.append(t)
            t = r
        else:
            t += r

    # Append the last gathered segment if it doesn't match the regex
    if not re.match(_split_regex, r):
        if t:
            new_res.append(t)
    else:
        # Append the remaining part to the last element in new_res
        new_res[-1] += r

    # Filter out any empty strings resulting from the split
    res = [r for r in res if len(r)]
    return new_res


# def split_is_ok(text):
#     text = text.strip()
#     text2 = split(text)

#     if text != "".join(text2):
#         return False

#     for i, (t1, t2) in enumerate(zip(text.split(), text2)):
#         if t1 != t2.strip():
#             return False

#     return True


def isupper(s: str) -> bool:
    """
    Check if the string contains any uppercase letters.

    Parameters
    ----------
    s : str
        The string to be checked for uppercase letters.

    Returns
    -------
    bool
        Returns True if the string contains any uppercase letters, False otherwise.
    """
    return re.search("[A-Z]", s) is not None


def char_span_to_word_span(
    char_span: Tuple[int, int], token_lens: List[int], tokens: List[str] = None
) -> Tuple[int, int]:
    """
    Convert character span to word span based on token lengths.

    Parameters
    ----------
    char_span : Tuple[int, int]
        The start and end indices of the character span.
    token_lens : List[int]
        The cumulative lengths of tokens.
    tokens : List[str], optional
        The list of tokens, by default None.

    Returns
    -------
    Tuple[int, int]
        The start and end indices of the word span.
    """
    start, end = char_span

    assert start >= 0, "Start index must be non-negative."
    assert start < end, "Start index must be less than end index."

    n = len(token_lens)

    # Find the closest word start indices using binary search
    word_start_1 = bisect.bisect_left(token_lens, start)
    word_start_2 = bisect.bisect_right(token_lens, start)

    word_start = None

    if word_start is None:
        # Calculate the distance to the closest word start
        e1 = abs(start - (token_lens[word_start_1 - 1] if word_start_1 > 0 else 0))
        e2 = abs(start - (token_lens[word_start_2 - 1] if word_start_2 > 0 else 0))

        word_start = word_start_1 if e1 < e2 else word_start_2

    # Find the closest word end indices using binary search
    word_end_1 = bisect.bisect_left(token_lens, end)
    word_end_2 = bisect.bisect_right(token_lens, end)

    e1 = abs(end - (token_lens[word_end_1] if word_end_1 < n else token_lens[-1]))
    e2 = abs(end - (token_lens[word_end_2] if word_end_2 < n else token_lens[-1]))

    word_end = word_end_1 if e1 < e2 else word_end_2
    word_end = min(word_end + 1, n)

    word_span = (word_start, word_end)

    return word_span


def get_word_ids_from_tokens(tokens: List[str]) -> np.ndarray:
    """
    Convert a list of tokens into word IDs. Each new word is assumed to start
    whenever a whitespace is detected in a token.

    Parameters
    ----------
    tokens : List[str]
        List of string tokens.

    Returns
    -------
    np.ndarray
        An array of word IDs corresponding to the tokens, with data type int16.

    Notes
    -----
    The function uses regular expression to check for whitespaces in tokens to
    determine the start of new words. Each word index starts from 0 and is
    incremented upon finding a whitespace.

    Examples
    --------
    >>> get_word_ids_from_tokens(['Hello', ' ', 'World'])
    array([0, 1, 1], dtype=int16)
    """
    # Initialize the list to store word indices
    word_ids = []
    # Index to track current word
    i = 0
    for t in tokens:
        # Increment word index if whitespace is found in the token
        if re.search("\s", t):
            i += 1
        # Append current word index to list
        word_ids.append(i)

    # Convert list of word indices to numpy array with type int16
    word_ids = np.array(word_ids, dtype=np.int16)
    return word_ids


def word_perc_pos_from_ids(word_ids: np.ndarray) -> np.ndarray:
    """
    Calculate the percentage-based position of each word in a sequence.

    Parameters
    ----------
    word_ids : np.ndarray
        An array of word IDs where consecutive identical IDs represent the same word.

    Returns
    -------
    np.ndarray
        An array of the same length as `word_ids` containing the position of each word
        expressed as a percentage of `cfg.MAXLEN`, rounded and converted to integers.

    Notes
    -----
    This function assumes the existence of a configuration object `cfg` with an attribute `MAXLEN`.
    The output values are clipped to a maximum of `cfg.MAXLEN - 1`.

    """
    # Calculate changes in word IDs to identify new words, starting from the second ID
    word_perc_pos = (word_ids[1:] != word_ids[:-1]).cumsum()
    # Insert a 0 at the beginning to align with the original length of `word_ids`
    word_perc_pos = np.insert(word_perc_pos, 0, 0)
    # Scale positions to the range of 0 to cfg.MAXLEN based on the max value found
    word_perc_pos = cfg.MAXLEN * word_perc_pos / (1 + word_perc_pos.max())
    # Ensure the positions do not exceed cfg.MAXLEN - 1
    word_perc_pos = np.minimum(word_perc_pos, cfg.MAXLEN - 1)
    # Round the positions and convert to integers
    word_perc_pos = word_perc_pos.round().astype(np.int64)

    return word_perc_pos


def gen_data_from_id(
    uuid: str, tokenizer, df: pd.DataFrame = None, root: str = None, text: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data from a given uuid and tokenizer.

    Parameters
    ----------
    uuid : str
        Unique identifier for the data.
    tokenizer : Any
        Tokenizer to convert text to tokens.
    df : pd.DataFrame, optional
        DataFrame containing discourse information. Default is None.
    root : str, optional
        Root directory to read the text file from. Default is None.
    text : str, optional
        Text to be tokenized. Default is None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns word_ids, word_perc_pos, token_sizes, input_ids, target, and eff_target arrays.
    """
    text = text or read_from_id(uuid, root=root).strip()
    nchars = len(text)
    tokens = split(text)
    word_ids = get_word_ids_from_tokens(tokens)
    offsets = np.cumsum([len(t) for t in tokens])
    ntokens = len(tokens)

    compute_target = df is not None

    target = None
    eff_target = None

    if compute_target:
        n_discourse_types = len(cfg.DISCOURSE2ID)
        data = df.loc[df["id"] == uuid]
        data = data[
            [
                "discourse_type_id",
                "discourse_start",
                "discourse_end",
                "discourse_effectiveness",
            ]
        ].values

        target = np.zeros(ntokens, dtype=np.int16)  # Target in Word dim
        eff_target = np.full(
            (ntokens,), fill_value=cfg.PYTORCH_CE_IGNORE_INDEX, dtype=np.int16
        )  # Effective target in Word dim

        for irow, row in enumerate(data):
            class_id, start, end, eff = row
            start, end = min(start, nchars), min(end, nchars)
            if start >= end:
                warn(f"The {irow}'th span for <{uuid}> is empty")
                continue

            start, end = char_span_to_word_span(
                (start, end), token_lens=offsets, tokens=tokens
            )

            if start < end:
                target[start] = class_id + n_discourse_types  # Beginning <B>
                eff_target[start:end] = cfg.EFF2ID[eff]

                if start < end - 1:
                    target[start + 1 : end] = class_id  # Inside <I>

    input_ids = tokenizer(tokens, add_special_tokens=False)["input_ids"]  # Word dim
    token_sizes = [len(t) for t in input_ids]  # Word dim

    token_sizes.insert(0, 0)  # Word dim + 1
    token_sizes = np.array(token_sizes, dtype=np.int16).cumsum()  # Word dim + 1

    word_ids = word_ids.repeat(token_sizes[1:] - token_sizes[:-1])
    word_perc_pos = word_perc_pos_from_ids(word_ids)
    input_ids = np.concatenate(input_ids)  # In Token dim

    if compute_target:
        target = target.repeat(
            token_sizes[1:] - token_sizes[:-1]
        )  # Repeat for words tokenized into several tokens => Token dim
        eff_target = eff_target.repeat(token_sizes[1:] - token_sizes[:-1])

        assert target.shape == input_ids.shape

        return word_ids, word_perc_pos, token_sizes, input_ids, target, eff_target
    else:
        return word_ids, word_perc_pos, token_sizes, input_ids


class Dataset(TorchDataset):
    """
    Torch dataset for training.

    Parameters
    ----------
    uuids : List[str]
        List of unique identifiers.
    data : dict
        Dictionary containing data indexed by uuid.
    pad_token_id : int
        ID of the pad token.
    mask_token_id : int
        ID of the mask token.
    maxlen : int, optional
        Maximum length of the sequence.
    is_train : bool, optional
        Flag indicating if the dataset is for training.
    p_mask_size : float, optional
        Probability mask size.
    p_mask_freq : float, optional
        Frequency of masking.
    special_token_ids : dict, optional
        Dictionary containing special token IDs.
    """

    def __init__(
        self,
        uuids: List[str],
        data: dict,
        pad_token_id: int,
        mask_token_id: int,
        maxlen: int = None,
        is_train: bool = True,
        p_mask_size: float = None,
        p_mask_freq: float = None,
        special_token_ids: dict = None,
    ):

        self.uuids = uuids
        self.data = data
        self.maxlen = maxlen or cfg.MAXLEN
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.is_train = is_train
        self.p_mask_freq = cfg.P_MASK_FREQ if p_mask_freq is None else p_mask_freq
        self.special_token_ids = special_token_ids

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.uuids)

    def mask_input_ids(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Masks a subset of input IDs.

        Parameters
        ----------
        input_ids : np.ndarray
            Array of input IDs.

        Returns
        -------
        np.ndarray
            Array of input IDs with some IDs masked.
        """
        p_mask_size = np.random.uniform(cfg.P_MASK_SIZE_LOW, cfg.P_MASK_SIZE_HIGH)
        n_masked = int(p_mask_size * len(input_ids))

        if n_masked > 0:
            index = np.random.choice(len(input_ids), size=n_masked, replace=False)
            input_ids[index] = self.mask_token_id

        return input_ids

    @property
    def get_special_tokens_pad_widths(self) -> Tuple[List[int], List[int]]:
        """
        Calculates padding widths and constant values for special tokens.

        Returns
        -------
        Tuple[List[int], List[int]]
            Pad widths and constant values for padding.
        """
        pad_width, constant_values = [0, 0], [-666, -666]
        if self.special_token_ids["bos_token_id"] is not None:
            pad_width[0] = 1
            constant_values[0] = self.special_token_ids["bos_token_id"]

        if self.special_token_ids["eos_token_id"] is not None:
            pad_width[1] = 1
            constant_values[1] = self.special_token_ids["eos_token_id"]

        return pad_width, constant_values

    def add_special_tokens(
        self, input_ids: np.ndarray, target: np.ndarray, eff_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Adds special tokens to input IDs, target, and effective target.

        Parameters
        ----------
        input_ids : np.ndarray
            Array of input IDs.
        target : np.ndarray
            Array of target IDs.
        eff_target : np.ndarray
            Array of effective target IDs.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Updated input IDs, target, and effective target with special tokens.
        """
        pad_width, constant_values = self.get_special_tokens_pad_widths

        if sum(pad_width):
            input_ids = np.pad(
                input_ids, pad_width=pad_width, constant_values=constant_values
            )

            target = np.pad(
                target, pad_width=pad_width, constant_values=cfg.PYTORCH_CE_IGNORE_INDEX
            )

            eff_target = np.pad(
                eff_target,
                pad_width=pad_width,
                constant_values=cfg.PYTORCH_CE_IGNORE_INDEX,
            )

        return input_ids, target, eff_target

    def truncate(
        self,
        word_perc_pos: np.ndarray,
        input_ids: np.ndarray,
        target: np.ndarray,
        eff_target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Truncates sequences to fit within the maximum length.

        Parameters
        ----------
        word_perc_pos : np.ndarray
            Array of word percentage positions.
        input_ids : np.ndarray
            Array of input IDs.
        target : np.ndarray
            Array of target IDs.
        eff_target : np.ndarray
            Array of effective target IDs.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Truncated and possibly masked input IDs, target, effective target, and masks.
        """
        input_ids = input_ids.astype(np.int64, copy=False)
        target = target.astype(np.int64, copy=False)
        eff_target = eff_target.astype(np.int64, copy=False)

        seq_len = self.maxlen - 2
        if self.is_train and np.random.random() < cfg.FORCE_TRUNC_FREQ:
            seq_len = min(seq_len, cfg.MIN_SEQ_LEN)

        if len(input_ids) > seq_len:
            if self.is_train:
                if np.random.random() < cfg.P_RANDOM_START:
                    start = np.random.choice(len(input_ids) - seq_len)
                elif np.random.random() < cfg.P_START_AT_SEQ_BEGINNING:
                    start = 0
                else:
                    start = len(input_ids) - seq_len
            else:
                start = 0

            input_ids = input_ids[start : start + seq_len]
            target = target[start : start + seq_len]
            eff_target = eff_target[start : start + seq_len]
            word_perc_pos = word_perc_pos[start : start + seq_len + 2]

        if self.is_train and np.random.rand() < self.p_mask_freq:
            input_ids = self.mask_input_ids(input_ids.copy())

        input_ids, target, eff_target = self.add_special_tokens(
            input_ids=input_ids, target=target, eff_target=eff_target
        )

        if len(word_perc_pos) < len(input_ids):
            word_perc_pos = np.concatenate(
                [
                    word_perc_pos,
                    [word_perc_pos[-1]] * (len(input_ids) - len(word_perc_pos)),
                ]
            )

        masks = np.ones_like(input_ids)

        return word_perc_pos, input_ids, masks, target, eff_target

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Processed sample containing word percentage positions, input IDs, masks, target, and effective target.
        """
        uuid = self.uuids[idx]
        word_ids, word_perc_pos, token_sizes, input_ids, target, eff_target = self.data[
            uuid
        ]

        return self.truncate(
            word_perc_pos=word_perc_pos,
            input_ids=input_ids,
            target=target,
            eff_target=eff_target,
        )


class TestDataset(Dataset):
    """
    Torch test dataset.

    Parameters
    ----------
    uuids : List[str]
        List of unique identifiers.
    pad_token_id : int
        Token ID for padding.
    mask_token_id : Optional[int], optional
        Token ID for masking, by default None.
    maxlen : Optional[int], optional
        Maximum length of sequences, by default None.
    stride : Optional[int], optional
        Stride length for truncation, by default None.
    is_train : bool, optional
        Indicator if the dataset is for training, by default False.
    special_token_ids : Optional[List[int]], optional
        List of special token IDs, by default None.
    data : Optional[Dict[str, Any]], optional
        Pre-loaded data, by default None.
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(
        self,
        uuids: List[str],
        pad_token_id: int,
        mask_token_id: Optional[int] = None,
        maxlen: Optional[int] = None,
        stride: Optional[int] = None,
        is_train: bool = False,
        special_token_ids: Optional[List[int]] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):

        self.uuids = uuids
        self.maxlen = maxlen or cfg.MAXLEN
        self.stride = stride or (
            self.maxlen // cfg.STRIDE_MAX_LEN_RATIO if self.maxlen < 1024 else 1024
        )
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.is_train = is_train
        self.special_token_ids = special_token_ids
        self.data = data
        self.kwargs = kwargs
        self.p_mask_freq = cfg.P_MASK_FREQ

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            Number of items in the dataset.
        """
        return len(self.uuids)

    def add_special_tokens(
        self,
        word_ids: np.ndarray,
        input_ids: np.ndarray,
        target: Optional[np.ndarray] = None,
        eff_target: Optional[np.ndarray] = None,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Adds special tokens to the input sequences.

        Parameters
        ----------
        word_ids : np.ndarray
            Array of word IDs.
        input_ids : np.ndarray
            Array of input IDs.
        target : Optional[np.ndarray], optional
            Target array, by default None.
        eff_target : Optional[np.ndarray], optional
            Effective target array, by default None.

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            Updated arrays with special tokens added.
        """
        pad_width, constant_values = self.get_special_tokens_pad_widths

        if sum(pad_width):
            word_ids = np.pad(word_ids, pad_width=pad_width, constant_values=-1)
            input_ids = np.pad(
                input_ids, pad_width=pad_width, constant_values=constant_values
            )
            if target is not None:
                target = np.pad(
                    target,
                    pad_width=pad_width,
                    constant_values=cfg.PYTORCH_CE_IGNORE_INDEX,
                )
                eff_target = np.pad(
                    eff_target,
                    pad_width=pad_width,
                    constant_values=cfg.PYTORCH_CE_IGNORE_INDEX,
                )

        return (
            (word_ids, input_ids, target, eff_target)
            if target is not None
            else (word_ids, input_ids)
        )

    def truncate(
        self,
        word_ids: np.ndarray,
        word_perc_pos: np.ndarray,
        input_ids: np.ndarray,
        target: Optional[np.ndarray] = None,
        eff_target: Optional[np.ndarray] = None,
    ) -> Union[
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]],
        Tuple[
            List[np.ndarray],
            List[np.ndarray],
            List[np.ndarray],
            List[np.ndarray],
            List[np.ndarray],
            List[np.ndarray],
        ],
    ]:
        """
        Truncates the input sequences to the maximum length with optional masking.

        Parameters
        ----------
        word_ids : np.ndarray
            Array of word IDs.
        word_perc_pos : np.ndarray
            Array of word percentage positions.
        input_ids : np.ndarray
            Array of input IDs.
        target : Optional[np.ndarray], optional
            Target array, by default None.
        eff_target : Optional[np.ndarray], optional
            Effective target array, by default None.

        Returns
        -------
        Union[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]],
              Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]]
            Truncated sequences and optionally targets.
        """
        if target is not None:
            target = target.astype(np.int64, copy=False)
            eff_target = eff_target.astype(np.int64, copy=False)

        input_ids_list = []
        masks_list = []
        token_ids_list = []
        word_perc_pos_list = []
        target_list = []
        eff_target_list = []

        if self.is_train and np.random.rand() < self.p_mask_freq:
            input_ids = self.mask_input_ids(input_ids.copy())

        for start in range(
            0, max(len(input_ids) - self.maxlen + 2, 0) + self.stride, self.stride
        ):
            end = min(start + self.maxlen - 2, len(input_ids))
            start = max(end - self.maxlen + 2, 0)

            word_id, input_id = word_ids[start:end], input_ids[start:end]
            if target is not None:
                t = target[start:end]
                eff_t = eff_target[start:end]
                word_id, input_id, t, eff_t = self.add_special_tokens(
                    word_ids=word_id,
                    input_ids=input_id,
                    target=t,
                    eff_target=eff_t,
                )
                target_list.append(t)
                eff_target_list.append(eff_t)
            else:
                word_id, input_id = self.add_special_tokens(
                    word_ids=word_id, input_ids=input_id
                )

            w = word_perc_pos[start : end + 2]
            if len(w) < len(input_id):
                w = np.concatenate([w, [w[-1]] * (len(input_id) - len(w))])

            word_perc_pos_list.append(w)
            input_ids_list.append(input_id)
            token_ids_list.append(word_id)
            masks_list.append(np.ones(len(input_id)))

        input_ids = input_ids_list
        masks = masks_list
        word_ids = token_ids_list
        word_perc_pos = word_perc_pos_list
        if target is not None:
            target = target_list
            eff_target = eff_target_list

        return (
            (word_ids, word_perc_pos, input_ids, masks)
            if target is None
            else (word_ids, word_perc_pos, input_ids, masks, target, eff_target)
        )

    def __getitem__(self, idx: int):
        """
        Retrieves an item from the dataset.

        Parameters
        ----------
        idx : int
            Index of the item.


        """
        uuid = self.uuids[idx]

        res = (
            gen_data_from_id(uuid, **self.kwargs)
            if self.data is None
            else self.data[uuid]
        )

        if len(res) == 4:
            word_ids, word_perc_pos, token_sizes, input_ids = res
            target = None
            eff_target = None
        else:
            word_ids, word_perc_pos, token_sizes, input_ids, target, eff_target = res

        return (
            idx,
            token_sizes,
            *self.truncate(
                word_ids=word_ids,
                word_perc_pos=word_perc_pos,
                input_ids=input_ids,
                target=target,
                eff_target=eff_target,
            ),
        )


class DynamicBatchDataset(TorchDataset):
    def __init__(self, dataset: TestDataset, batch_size: int, sizes: List[int]):
        """
        Initialize the DynamicBatchDataset.

        Parameters
        ----------
        dataset : TestDataset
            The dataset from which to create dynamic batches.
        batch_size : int
            The maximum size of each batch.
        sizes : List[int]
            A list of sizes representing the size of each data point in the dataset.
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.sizes = sizes
        self.spans = self.get_batch_spans()

    def get_batch_spans(self) -> List[Tuple[int, int]]:
        """
        Calculate the spans for each batch based on the sizes and batch size.

        Returns
        -------
        List[Tuple[int, int]]
            A list of tuples where each tuple contains the start and end indices for a batch.
        """
        sizes = self.sizes
        batch_size = self.batch_size

        if len(sizes) < 2:
            return [(0, len(sizes))]

        spans = []
        s = 0
        i = 0
        start = 0
        while i < len(sizes):
            s += sizes[i]

            if s > batch_size:
                end = max(start + 1, i)
                spans.append((start, end))
                i = end
                start = i
                s = 0
            else:
                i += 1

        if not spans or spans[-1][1] < len(sizes):
            spans.append((start, len(sizes)))
        return spans

    def __len__(self) -> int:
        """
        Get the number of batches.

        Returns
        -------
        int
            The number of batches.
        """
        return len(self.spans)

    def __getitem__(self, idx: int) -> List[Any]:
        """
        Get the samples for a specific batch.

        Parameters
        ----------
        idx : int
            The index of the batch to retrieve.

        Returns
        -------
        List[Any]
            A list of samples for the specified batch.
        """
        span = self.spans[idx]
        samples = []
        for idx in range(span[0], span[1]):
            samples.append(self.dataset[idx])

        return samples


def pad_to_bach_maxlen(
    batch_ids: List[int],
    token_ids: List[List[List[int]]],
    word_perc_pos: List[List[List[int]]],
    input_ids: List[List[List[int]]],
    masks: List[List[List[int]]],
    pad_token_id: int,
    target: Optional[List[List[List[int]]]] = None,
    eff_target: Optional[List[List[List[int]]]] = None,
    max_item: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Pads the input batch to the maximum sequence length within the batch.

    Parameters
    ----------
    batch_ids : List[int]
        List of batch IDs.
    token_ids : List[List[List[int]]]
        List of token IDs for each item in the batch.
    word_perc_pos : List[List[List[int]]]
        List of word percentage positions for each item in the batch.
    input_ids : List[List[List[int]]]
        List of input IDs for each item in the batch.
    masks : List[List[List[int]]]
        List of masks for each item in the batch.
    pad_token_id : int
        The padding token ID.
    target : Optional[List[List[List[int]]]], optional
        Target labels for each item in the batch, by default None.
    eff_target : Optional[List[List[List[int]]]], optional
        Effective target labels for each item in the batch, by default None.
    max_item : Optional[int], optional
        Maximum number of items to consider from each list, by default None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing padded batch IDs, token IDs, word percentage positions, input IDs, masks,
        and optionally targets and effective targets if provided.
    """

    # Determine the size of each batch item
    batch_item_sizes = [
        len(x) if max_item is None else min(len(x), max_item) for x in token_ids
    ]

    # Calculate total batch size and maximum sequence length
    batch_size = sum(batch_item_sizes)
    batch_maxlen = max([len(xx) for x in token_ids for xx in x])

    # Repeat batch IDs to match the new batch size
    batch_ids = np.array(batch_ids).repeat(batch_item_sizes)

    # Initialize padded arrays
    shape = (batch_size, batch_maxlen)
    new_token_ids = np.full(shape, -1, dtype=np.int16)
    new_word_perc_pos = np.zeros(shape, dtype=np.int64)
    new_input_ids = np.full(shape, pad_token_id, dtype=np.int64)
    new_masks = np.zeros(shape, dtype=np.int64)

    # Initialize target arrays if target is provided
    compute_target = target is not None
    if compute_target:
        new_target = np.full(shape, cfg.PYTORCH_CE_IGNORE_INDEX, dtype=np.int64)
        new_eff_target = np.full(shape, cfg.PYTORCH_CE_IGNORE_INDEX, dtype=np.int64)
    else:
        target = [[[]] * size for size in batch_item_sizes]
        eff_target = [[[]] * size for size in batch_item_sizes]

    pos = 0
    # Iterate over all elements and pad them accordingly
    for (
        token_id_list,
        word_perc_list,
        input_id_list,
        mask_list,
        target_list,
        eff_target_list,
    ) in zip(token_ids, word_perc_pos, input_ids, masks, target, eff_target):
        for i, (token_id, word_perc, input_id, mask, t, eff_t) in enumerate(
            zip(
                token_id_list,
                word_perc_list,
                input_id_list,
                mask_list,
                target_list,
                eff_target_list,
            )
        ):

            if max_item is not None and i >= max_item:
                break

            # Fill new arrays with the current item data
            new_token_ids[pos, : len(token_id)] = token_id
            new_word_perc_pos[pos, : len(word_perc)] = word_perc
            new_input_ids[pos, : len(input_id)] = input_id
            new_masks[pos, : len(mask)] = mask

            if compute_target:
                new_target[pos, : len(t)] = t
                new_eff_target[pos, : len(eff_t)] = eff_t

            pos += 1

    return (
        batch_ids,
        new_token_ids,
        new_word_perc_pos,
        new_input_ids,
        new_masks,
        *((new_target, new_eff_target) if compute_target else ()),
    )


def collate_fn(
    inputs: List[
        Tuple[
            List[List[int]],
            List[List[int]],
            List[List[int]],
            List[List[int]],
            List[List[int]],
            List[List[int]],
            Optional[List[List[int]]],
            Optional[List[List[int]]],
        ]
    ],
    pad_token_id: int,
    max_item: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Collates inputs into batches, padding them to the same length.

    Parameters
    ----------
    inputs :
        A list of tuples containing input data.
    pad_token_id : int
        Token ID used for padding.
    max_item : Optional[int], optional
        Maximum number of items to include in the batch, by default None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing padded batch IDs, token IDs, word percentage positions, input IDs, masks,
        and optionally targets and effective targets if provided.
    """
    inputs = tuple(zip(*inputs))

    assert len(inputs) in [6, 8], f"{len(inputs)} is an Unknown number of elements"

    # Unpack inputs
    batch_ids, token_pos, token_ids, word_perc_pos, input_ids, masks = inputs[:6]
    target, eff_target = inputs[6:] if len(inputs) >= 7 else (None, None)

    return pad_to_bach_maxlen(
        batch_ids=batch_ids,
        token_ids=token_ids,
        word_perc_pos=word_perc_pos,
        input_ids=input_ids,
        masks=masks,
        target=target,
        eff_target=eff_target,
        pad_token_id=pad_token_id,
        max_item=max_item,
    )


def collate_fn_train(
    inputs: List[Tuple[List[int], List[int], List[int], List[int], List[int]]],
    pad_token_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collate function for training data.

    Parameters
    ----------
    inputs :
        Batch of input data, each containing word_perc_pos, input_ids, masks, target, and eff_target.
    pad_token_id : int
        Padding token ID.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Padded and collated word_perc_pos, input_ids, masks, target, and eff_target arrays.
    """
    # Unzip the inputs into separate lists
    inputs = tuple(zip(*inputs))

    # Ensure the correct number of input elements
    assert len(inputs) == 4, f"{len(inputs)} is an Unknown number of elements"

    word_perc_pos, input_ids, masks, target, eff_target = inputs

    batch_size = len(input_ids)
    batch_maxlen = max([len(x) for x in input_ids])

    # Define the shape of the new arrays
    shape = (batch_size, batch_maxlen)

    # Initialize new arrays with default values
    new_word_perc_pos = np.full(shape, -1, dtype=np.int64)
    new_input_ids = np.full(shape, pad_token_id, dtype=np.int64)
    new_masks = np.zeros(shape, dtype=np.int64)
    new_target = np.full(shape, cfg.PYTORCH_CE_IGNORE_INDEX, dtype=np.int64)
    new_eff_target = np.full(shape, cfg.PYTORCH_CE_IGNORE_INDEX, dtype=np.int64)

    # Populate the new arrays with data from the batch
    for pos, (word_id, input_id, mask, t, eff_t) in enumerate(
        zip(word_perc_pos, input_ids, masks, target, eff_target)
    ):
        new_word_perc_pos[pos, : len(word_id)] = word_id
        new_input_ids[pos, : len(input_id)] = input_id
        new_masks[pos, : len(mask)] = mask
        new_target[pos, : len(t)] = t
        new_eff_target[pos, : len(eff_t)] = eff_t

    return new_word_perc_pos, new_input_ids, new_masks, new_target, new_eff_target


def collate_fn_list(inputs, pad_token_id, max_item=None):
    return collate_fn(it_chain(*inputs), pad_token_id=pad_token_id, max_item=max_item)


class RunningMax:
    """
    Will track the max for each position over running windows.
    """

    def __init__(self):
        """
        Initializes the RunningMax instance with empty values and lists.
        """
        self.vals: pd.DataFrame = None
        self._ids: List[np.ndarray] = []
        self._vals: List[np.ndarray] = []

    def append(self, ids: np.ndarray, vals: np.ndarray, reduce: bool = False) -> 'RunningMax':
        """
        Appends new ids and values to the running lists.

        Parameters
        ----------
        ids : np.ndarray
            The ids to append.
        vals : np.ndarray
            The values to append.
        reduce : bool, optional
            Flag to indicate if reduction should be performed after appending, by default False.

        Returns
        -------
        RunningMax
            The instance itself for chaining.
        """
        self._ids.append(ids)
        self._vals.append(vals)
        return self

    def reset(self) -> 'RunningMax':
        """
        Resets the internal ids and values lists.

        Returns
        -------
        RunningMax
            The instance itself for chaining.
        """
        self._ids = []
        self._vals = []
        return self

    def reduce(self) -> 'RunningMax':
        """
        Reduces the collected ids and values, computing the max for each position.

        Returns
        -------
        RunningMax
            The instance itself for chaining.
        """
        ids = np.concatenate(self._ids)
        vals = np.concatenate(self._vals)

        vals = pd.DataFrame(vals, index=tuple(zip(*ids.T)))

        if self.vals is not None:
            vals = pd.concat([self.vals, vals], axis=0, sort=False)

        self.vals = vals.groupby(level=(0, 1)).max()

        self.reset()

        return self

    def update(self, ids: np.ndarray, vals: np.ndarray, reduce: bool = False) -> 'RunningMax':
        """
        Updates the instance with new ids and values, optionally reducing.

        Parameters
        ----------
        ids : np.ndarray
            The ids to update.
        vals : np.ndarray
            The values to update.
        reduce : bool, optional
            Flag to indicate if reduction should be performed after updating, by default False.

        Returns
        -------
        RunningMax
            The instance itself for chaining.
        """
        self.append(ids, vals)
        if reduce:
            self.reduce()
        return self

    def normlaize(self) -> 'RunningMax':
        """
        Normalizes the values by dividing each by the sum of its row.

        Returns
        -------
        RunningMax
            The instance itself for chaining.
        """
        self.vals /= self.vals.sum(1).values[:, None]
        return self

