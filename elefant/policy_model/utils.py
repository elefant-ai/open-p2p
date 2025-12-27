import logging
import fsspec
import pydantic_yaml
import os

from elefant.config import ConfigBase, load_config
from elefant.data.action_mapping import UniversalAutoregressiveActionMapping
from elefant.policy_model.config import LightningPolicyConfig

import torch
import torch.nn as nn


class ActionDimensionConverter(nn.Module):
    def __init__(self, num_actions_per_dim: torch.Tensor):
        super().__init__()
        assert num_actions_per_dim.max() == num_actions_per_dim.min(), (
            "Support only for same number of actions per dimension when calculating true action RCE"
        )
        self.num_actions_per_dim = num_actions_per_dim

        if len(num_actions_per_dim) > 1:
            num_actions = num_actions_per_dim[0]
            powers = torch.tensor(
                [num_actions**i for i in reversed(range(len(num_actions_per_dim)))]
            )

            self.register_buffer("powers", powers, persistent=False)

    @torch.compile(mode="reduce-overhead")
    def forward(self, action_per_dim: torch.Tensor) -> torch.Tensor:
        if len(self.num_actions_per_dim) == 1:
            return action_per_dim

        true_action = action_per_dim * self.powers
        return true_action.sum(dim=-1)

    def inverse(self, true_action: torch.Tensor) -> torch.Tensor:
        """Convert from integer representation back to vector representation.

        Args:
            true_action: Integer representation of the action (B,) or scalar

        Returns:
            Vector representation of the action (B, D) or (D,)
        """
        if len(self.num_actions_per_dim) == 1:
            return (
                true_action.unsqueeze(-1)
                if true_action.dim() > 0
                else true_action.unsqueeze(0)
            )

        num_actions = self.num_actions_per_dim[0]
        batch_size = true_action.shape[0] if true_action.dim() > 0 else 1
        true_action = true_action.view(-1)

        # Initialize output tensor
        action_per_dim = torch.zeros(
            (batch_size, len(self.num_actions_per_dim)),
            device=true_action.device,
            dtype=torch.long,
        )

        # Convert each integer back to vector representation
        for i in range(len(self.num_actions_per_dim)):
            action_per_dim[:, i] = (
                true_action // (num_actions ** (len(self.num_actions_per_dim) - 1 - i))
            ) % num_actions

        return action_per_dim.squeeze(0) if batch_size == 1 else action_per_dim

    @staticmethod
    def test_conversion():
        """Test function to verify the conversion and inverse conversion work correctly."""
        # Test case 1: Single dimension
        num_actions = torch.tensor([5])
        converter = ActionDimensionConverter(num_actions)
        test_action = torch.tensor([3])
        true_action = converter(test_action)
        recovered_action = converter.inverse(true_action)
        assert torch.all(test_action == recovered_action), (
            "Single dimension test failed"
        )

        # Test case 2: Multiple dimensions
        num_actions = torch.tensor([3, 3, 3])
        converter = ActionDimensionConverter(num_actions)
        test_action = torch.tensor([[1, 2, 0], [0, 1, 2]])  # Batch of 2 actions
        true_action = converter(test_action)
        recovered_action = converter.inverse(true_action)
        assert torch.all(test_action == recovered_action), (
            "Multiple dimensions test failed"
        )

        # Test case 3: Single action in multiple dimensions
        test_action = torch.tensor([1, 2, 0])
        true_action = converter(test_action)
        recovered_action = converter.inverse(true_action)
        assert torch.all(test_action == recovered_action), "Single action test failed"

        print("All conversion tests passed!")


def load_config_from_checkpoint(checkpoint_path: str) -> LightningPolicyConfig:
    # The checkpoint path can either be a folder or a file.
    checkpoint_folder = os.path.dirname(checkpoint_path)
    specific_checkpoint = checkpoint_path
    config_path = checkpoint_folder.rstrip("/") + "/model_config.yaml"
    logging.info(f"Loading config from {config_path}")
    config = load_config(config_path, LightningPolicyConfig)

    logging.info(f"Using checkpoint {specific_checkpoint}")
    config.inference.checkpoint_path = specific_checkpoint
    return config
