from torch import tensor, float32 as t_f32

import abc


class RunningFractionalMetric(object):
    def __init__(self, numerator: tensor, denominator: tensor, smooth: float=1e-6) -> None:
        self.numerator: tensor = numerator
        self.denominator: tensor = denominator
        self.smooth: float = smooth

    # metric: tensor
    # count: tensor
    def add_nums_and_denoms(
            self, numerator: tensor,
            denominator: tensor) -> None:
        self.numerator += numerator
        self.denominator += denominator

    def finalize(self) -> tensor:
        return (self.numerator + self.smooth) / (self.denominator + self.smooth)


class BatchedMetric(RunningFractionalMetric):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args) -> None:
        super().__init__(*args)

    # Function to be called to add batch
    def add_batch(self, preds: tensor, labels: tensor) -> None:
        nums, denoms = self.compute_batch_metric(preds, labels)
        self.add_nums_and_denoms(nums, denoms)

    # Function to be called to compute across all batches
    @abc.abstractmethod
    def compute(self):
        return None

    @abc.abstractmethod
    # Returns torch tensor
    def compute_batch_metric(self, preds, labels):
        return None
