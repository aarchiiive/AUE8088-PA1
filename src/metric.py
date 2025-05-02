from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError("Predictions and targets must have the same shape.")

        # Convert to boolean tensors
        tp = (preds == 1) & (target == 1)
        fp = (preds == 1) & (target == 0)
        fn = (preds == 0) & (target == 1)

        # Accumulate
        self.true_positives += tp.sum()
        self.false_positives += fp.sum()
        self.false_negatives += fn.sum()

    def compute(self):
        precision = self.true_positives.float() / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives.float() / (self.true_positives + self.false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError("Predictions and targets must have the same shape.")

        # [TODO] Count the number of correct prediction
        correct = (preds == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()