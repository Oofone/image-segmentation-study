from metrics.metrics import BatchedMetric

from torch import tensor, zeros, argmax, float32 as t_f32, device

import abc


class MultiClassSegmentationMetric(BatchedMetric):
    __metaclass__ = abc.ABCMeta

    def __init__(self, num_classes, device = device('cpu')):
        super(BatchedMetric, self).__init__(
            zeros(num_classes, dtype=t_f32, device=device),
            zeros(num_classes, dtype=t_f32, device=device)
        )
        self.num_classes = num_classes

    def compute(self):
        classwise = self.finalize()
        return classwise.mean().item()


class MeanPixelAccuracy(MultiClassSegmentationMetric):
    def __init__(self, num_classes, device = device('cpu')):
        super().__init__(num_classes, device)
        self.device = device

    def compute_batch_metric(self, preds, labels):
        # Convert predicted probabilities to class indices
        # Input: [B, N_class, H, W]
        preds = argmax(preds, dim=1)  # Output Shape: [B, H, W]

        # Initialize variables to store correct and totals
        correct_preds: tensor = zeros(self.num_classes, dtype=t_f32, device=self.device)
        totals: tensor = zeros(self.num_classes, dtype=t_f32, device=self.device)

        for cls in range(self.num_classes):
            # True positives (intersection): both preds and target are of class `cls`
            correct_preds[cls] = (preds == cls).float().sum().item() # Sum over H, W for each image
            
            # Union: pixels that are either in preds or targets as class `cls`
            totals[cls] = (labels == cls).float().sum().item()  # Sum over H, W for each image
        
        return correct_preds, totals


class MeanIoU(MultiClassSegmentationMetric):
    def __init__(self, num_classes, device = device('cpu')):
        super().__init__(num_classes, device)
        self.device = device

    def compute_batch_metric(self, preds, labels):
        # Convert predicted probabilities to class indices
        # Input: [B, N_class, H, W]
        preds = argmax(preds, dim=1)  # Output Shape: [B, H, W]

        # Initialize variables to store intersection and union
        intersections = zeros(self.num_classes, dtype=t_f32, device=self.device)
        unions = zeros(self.num_classes, dtype=t_f32, device=self.device)
                
        for cls in range(self.num_classes):
            # True positives (intersection): both preds and target are of class `cls`
            intersections[cls] = ((preds == cls) & (labels == cls)).float().sum().item() # Sum over H, W for each image
            
            # Union: pixels that are either in preds or targets as class `cls`
            unions[cls] = ((preds == cls) | (labels == cls)).float().sum().item()  # Sum over H, W for each image
        
        return intersections, unions


class MeanF1Score(MultiClassSegmentationMetric):
    def __init__(self, num_classes, device = device('cpu')):
        super().__init__(num_classes, device)
        self.device = device

    def compute_batch_metric(self, preds, labels):
        # Convert predicted probabilities to class indices
        # Input: [B, N_class, H, W]
        preds = argmax(preds, dim=1)  # Output Shape: [B, H, W]

        # Initialize variables to store intersection and union
        precisions = zeros(self.num_classes, dtype=t_f32, device=self.device)
        recalls = zeros(self.num_classes, dtype=t_f32, device=self.device)

        for cls in range(self.num_classes):
            # True positives (Sum over H, W for each image):
            # both preds and target are of class `cls`
            true_positives = ((preds == cls) & (labels == cls)).float().sum()

            # False positives (Sum over H, W for each image):
            # preds are of class 'cls' and targets are NOT
            false_positives = ((preds == cls) & (labels != cls)).float().sum()

            # False negatives (Sum over H, W for each image):
            # preds are NOT of class 'cls' and targets are of class 'cls'
            false_negatives = ((preds != cls) & (labels == cls)).float().sum()

            # Precision = TP / (TP + FP)
            # Smoothing = 1e-6 to avoid division by zero
            precisions[cls] = (true_positives / (true_positives + false_positives + 1e-6)).item()

            # Recall = TP / (TP + FN)
            # Smoothing = 1e-6 to avoid division by zero
            recalls[cls] = (true_positives / (true_positives + false_negatives + 1e-6)).item()

        # F1 score numerator:
        # 2 * precision * recall
        numerators = 2.0 * precisions * recalls

        # F1 score denominator:
        # precision + recall + 1e-6 (avoid divide by zero)
        denominators = precisions + recalls + 1e-6

        return numerators, denominators
