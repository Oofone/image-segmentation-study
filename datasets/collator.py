from typing import Tuple, List, Dict, Any
from torch import Tensor, stack, tensor


class BatchCollaterMappilaryVistasPTLightning(object):

    def __init__(self, has_label: bool = True):
        self.has_label = has_label

    def __call__(self, batch: List[Dict[str, Any]], class_stats: bool = False) -> Tuple[Tensor]:
        names = [x["name"] for x in batch]
        x_batch = stack([x['image'] for x in batch], dim=0)
        y_batch = y_labs = y_eval_labs = None

        if self.has_label:
            y_batch = stack([x['target'] for x in batch], dim=0)
            if class_stats:
                y_labs = [x['target_labels'] for x in batch]
                y_eval_labs = [x['target_labels_in_eval'] for x in batch]

        return {
            'names': names,
            'X': x_batch,
            'Y': y_batch,
            'Y_labs': y_labs,
            'Y_eval_labs': y_eval_labs,
        }
