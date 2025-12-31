import torch
from torchmetrics import Metric


# For reporting losses correctly when using gradient accumulation.
# see: https://github.com/Lightning-AI/pytorch-lightning/discussions/8682
class LossMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "loss", torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum"
        )
        self.add_state(
            "counter", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )

    def update(self, loss):
        self.loss += loss
        self.counter += 1

    def compute(self):
        return self.loss / self.counter

    def reset(self):
        self.loss.zero_()
        self.counter.zero_()
