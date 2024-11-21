
from typing import List, Literal
from enum import Enum


CHECKPOINT_PATH = "ckpt"
DEFAULT_LOGS_FILE = "logs.out"

BINARY_MODE = "BINARY_MODE"
MULTICLASS_MODE = "MULTICLASS_MODE"
MULTILABEL_MODE = "MULTILABEL_MODE"

class ModelType(Enum):
    Unet = "Unet"
    SegFormerHF = "SegFormerHF"
    SegFormer = "SegFormer"

    def _values() -> List[str]:
        return list(map(lambda x: str(x.value), ModelType))

_MODEL_TYPE_PARAM = Literal["Unet", "SegFormerHF", "SegFormer"]

class LossFunction(Enum):
    CrossEntropyLoss = "CrossEntropyLoss"
    FocalLoss = "FocalLoss"

    def _values() -> List[str]:
        return list(map(lambda x: str(x.value), LossFunction))

_LOSS_FUNCTION_PARAM = Literal["CrossEntropyLoss", "FocalLoss"]