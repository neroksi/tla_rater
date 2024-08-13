import logging
import os
from logging import config as lg_cfg
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import logging as tr_logging

tr_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fast DEBUG mode
IS_DEBUG = False
NUM_DEBUG_ROWS = 50

# Reproducibility
SEED = 321

EFF2ID = {"Effective": 1, "Non-Effective": 0}
ID2EFF = {v: k for k, v in EFF2ID.items()}

DISCOURSE2ID = {
    "Claim": 1,
    "Concluding Statement": 2,
    "Counterclaim": 3,
    "Evidence": 4,
    "Lead": 5,
    "Position": 6,
    "Rebuttal": 7,
}
ID2DISCOURSE = {id_: discourse for discourse, id_ in DISCOURSE2ID.items()}


# Model Validation Strategy
FOLD_COL_NAME = "fold"
MAIN_METRIC_NAME = "iov_v2_val"
FOLDS = [0, 1, 2, 3, 4]


# Folders & Paths
# fmt: off
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
TRAIN_ROOT = DATA_ROOT / "essays"  # full essay contents folder
TRAIN_CSV_PATH = DATA_ROOT / "train_v2.csv" # training data csv file, in the correct format
FOLD_JSON_PATH = DATA_ROOT / "fold_dict.json" # the folds, could be created using dataset.add_training_fold(...)
MODEL_ROOT = PROJECT_ROOT / "models" # model checkpointing root folder
LOGGING_CONFIG_YAML_PATH = PROJECT_ROOT / "logging.yaml" # python logger configuration yaml

LOGS_TXT_PATH = "logs.txt" # the logging file, will contain log messages

# fmt: on

# Modeling & Training
MODEL_NAME = "roberta-base"
NUM_PURE_TARGETS = len(DISCOURSE2ID)
NUM_INTERVAL = 2
NUM_TARGETS = 1 + NUM_PURE_TARGETS * NUM_INTERVAL
MAXLEN = 512

CLASS_WEIGHTS: list = None  # will be set in init_config()
SEG_CLASS_WEIGHTS: list = None  # will be set in init_config()

USE_AMP = True  # Enable AMP for fast fp16 training ?
EPOCHS = 5

DEVICE = torch.device("cpu")
CLIP_GRAD_NORM = 2.0
EARLY_STOP_EPOCH = 4
SAVE_EACH = 0.2499
TRUE_SEG_COEF = 0.60

TRAIN_BATCH_SIZE = 2
TRAIN_NUM_WORKERS = 0
VAL_BATCH_SIZE = 2
VAL_NUM_WORKERS = 0

P_DROPS = None
N_DROPS = 5

# Will initialize model weights with these weights => {fold_i: "path/to/ckpt.pth"}
CHECKPOINT_PATHS = {}

# Loss & Optimizer
OPTIMIZER_LR = 5e-6
OPTIMIZER_WEIGHT_DECAY = 0.01  # 1e-5
SCHEDULER_ETA_MIN = 8e-7

ALPHA_NER = 0.20  # Weight of pure NER objective in loss computation
ALPHA_SEG = 0.80  # Weight of pure SEGmentation objective in loss computation
ALPHA_EFF = 0.0  # Weight of pure EFFectiveness objective in loss computation

# Efficient Data Usage Strategy
P_MASK_SIZE_LOW = 0.15
P_MASK_SIZE_HIGH = 0.35
P_MASK_FREQ = 0.0  # 0.80 #0.80

P_RANDOM_START = 0.0  # 0.50
P_START_AT_SEQ_BEGINNING = 1.0  # 0.80 # Prob to start at beginning if not random start
MIN_SEQ_LEN = 4096  # 512 # All sequences longer than this could be truncated
FORCE_TRUNC_FREQ = 0.50

PYTORCH_CE_IGNORE_INDEX = -100
STRIDE_MAX_LEN_RATIO = 2
# POSITION_DIV_FACTOR = 3


DISCOURSE2WEIGHTS = {
    "Claim": 0.127,
    "Concluding Statement": 0.075,
    "Counterclaim": 0.182,
    "Evidence": 0.102,
    "Lead": 0.079,
    "Position": 0.102,
    "Rebuttal": 0.259,
}


ID2WEIGHTS = {
    1: 0.20,
    2: 0.80,
}


def to_1_7(x):
    if x == 0:
        return 0
    return 1 + (x - 1) % NUM_PURE_TARGETS


def init_config():
    global logger, NUM_TARGETS, NUM_PURE_TARGETS, CLASS_WEIGHTS, SEG_CLASS_WEIGHTS

    NUM_PURE_TARGETS = len(DISCOURSE2ID)
    NUM_TARGETS = 1 + NUM_PURE_TARGETS * NUM_INTERVAL

    CLASS_WEIGHTS = np.zeros(NUM_TARGETS, dtype=np.float32)

    for i in range(1, NUM_TARGETS):
        if i == 0:
            pos = 0
        elif i < 8:
            pos = 1
        elif i < 15:
            pos = 2
        else:
            pos = 3

        CLASS_WEIGHTS[i] = ID2WEIGHTS[pos] * DISCOURSE2WEIGHTS[ID2DISCOURSE[to_1_7(i)]]

    CLASS_WEIGHTS[0] = 0.50 * (1 - sum(DISCOURSE2WEIGHTS.values()))
    CLASS_WEIGHTS /= CLASS_WEIGHTS.sum()
    CLASS_WEIGHTS *= NUM_TARGETS

    SEG_CLASS_WEIGHTS = 3 * np.array([0.35, 0.25, 0.40], dtype=np.float32)

    with open(LOGGING_CONFIG_YAML_PATH, "r") as f:
        logging_config = yaml.load(f, Loader=yaml.FullLoader)

    if not logging_config["handlers"]["file"]["filename"]:
        logging_config["handlers"]["file"]["filename"] = LOGS_TXT_PATH

    logging_config["handlers"]["file"]["filename"] = str(
        Path(logging_config["handlers"]["file"]["filename"]).resolve().absolute()
    )

    lg_cfg.dictConfig(logging_config)
    logger = logging.getLogger("raterLogger")


try:
    init_config()
except Exception as e:
    print(f"INIT ERROR: {e}")
