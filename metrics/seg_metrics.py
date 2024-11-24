from metrics.metrics import BatchedMetric

from torch import zeros, argmax, numel, sum, float32 as t_f32, device

import abc


class SegmentationMetric(BatchedMetric):
    __metaclass__ = abc.ABCMeta

    def __init__(self, device = device('cpu')):
        super().__init__(
            zeros(1, dtype=t_f32, device=device),
            zeros(1, dtype=t_f32, device=device)
        )

    def compute(self):
        return self.finalize().item()


class DiceCoefficient(SegmentationMetric):
    def __init__(self, device = device('cpu')):
        super().__init__(device=device)
        self.device = device

    def compute_batch_metric(self, preds, labels):
        # Input: [B, N_class, H, W]
        preds = argmax(preds, dim=1)  # Output Shape: [B, H, W]

        # Only compare non-background classes (0)
        intersection = sum((preds == labels) * (labels > 0))
        dice_num = (2. * intersection).to(self.device)
        dice_denom = (sum(preds > 0) + sum(labels > 0)).to(self.device)

        # Dice = (2 x intersection) / (preds_size + labels_size)
        return dice_num, dice_denom
