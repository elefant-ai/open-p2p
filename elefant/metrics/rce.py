from torchmetrics import Metric
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F


class RCE(Metric):
    """Calculate the RCE of multivariate situation.

    Args:
        num_classes (Optional[int]): Number of classes if known beforehand. If None,
            will dynamically expand as new labels are encountered.
        epsilon: avoid underflow
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.num_classes = num_classes

        # Initialize state for counting labels
        if num_classes is not None:
            self.add_state(
                "label_counts",
                default=torch.zeros(num_classes, dtype=torch.long),
                dist_reduce_fx="sum",
            )
        else:
            self.add_state(
                "label_counts",
                default=torch.zeros(0, dtype=torch.long),
                dist_reduce_fx="sum",
            )

        # Keep track of total samples
        self.add_state(
            "total_samples",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "cumulative_step",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "cumulative_baseline_cross_entropy",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "validation_loss", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def reset_metric(self) -> None:
        print("resetting rce metrics")
        self.cumulative_step = 0
        self.cumulative_baseline_cross_entropy = 0
        self.validation_loss = 0

    def update(self, labels: torch.Tensor) -> None:
        """Update state with new batch of labels.

        Args:
            labels (torch.Tensor): Tensor of labels of shape (N,) where N is batch size
        """
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        # Ensure labels are long/int type
        labels = labels.long()

        # If num_classes wasn't specified, dynamically expand state as needed
        if self.num_classes is None:
            max_label = labels.max().item()
            if max_label >= len(self.label_counts):
                # Expand label_counts tensor to accommodate new labels
                new_size = max_label + 1
                new_counts = torch.zeros(
                    new_size, dtype=torch.long, device=self.label_counts.device
                )
                new_counts[: len(self.label_counts)] = self.label_counts
                self.label_counts = new_counts

        # Update counts
        unique_labels, counts = torch.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            self.label_counts[label] += count
        self.total_samples += len(labels)

    def update_metric(
        self, labels: torch.Tensor, validation_loss: torch.Tensor
    ) -> None:
        if not isinstance(validation_loss, torch.Tensor):
            validation_loss = torch.tensor(validation_loss)
        current_dist = self.label_counts.float() + self.epsilon
        current_dist = current_dist / current_dist.sum()

        log_probs = torch.log(current_dist)
        batch_cross_entropy = F.nll_loss(
            log_probs.unsqueeze(0).repeat(len(labels), 1), labels
        )
        self.cumulative_baseline_cross_entropy += batch_cross_entropy
        self.validation_loss += validation_loss
        self.cumulative_step += 1

    def compute(self) -> Dict[str, Any]:
        """Compute the label distribution metrics.

        Returns:
            Dict containing:
                - 'distribution': Tensor of label ratios
                - 'counts': Raw counts of each label
                - 'total_samples': Total number of samples seen
        """
        # Compute ratios
        assert self.cumulative_step > 0, (
            "need to update cross entropy before compute it"
        )
        avg_baseline_cross_entropy = (
            self.cumulative_baseline_cross_entropy.float() / self.cumulative_step
        )
        avg_validation_loss = self.validation_loss.float() / self.cumulative_step
        rce = (
            100
            * (avg_baseline_cross_entropy - avg_validation_loss)
            / avg_baseline_cross_entropy
        )
        ## clip at -1 to make nicer visualization
        rce = torch.clamp(rce, min=-1.0)

        self.reset_metric()
        return rce, avg_validation_loss
