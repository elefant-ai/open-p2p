import torch
from elefant.config import ConfigBase
import logging
from torch.utils.data import DataLoader
from elefant.data.action_label_video_proto_dataset import ActionLabelVideoDatasetItem
from elefant.data.action_mapping import UniversalAutoregressiveActionMapping
from elefant.data.action_mapping import UniversalAutoregressiveActionMappingConfig


class DummyDatasetConfig(ConfigBase):
    """Dataset that just returns random tensor indefinitely."""

    frame_height: int = 192
    frame_width: int = 192
    T: int = 10
    action_mapping: UniversalAutoregressiveActionMappingConfig


class DummyDataset(torch.utils.data.IterableDataset):
    """Dataset that just returns random tensor indefinitely."""

    def __init__(self, cfg: DummyDatasetConfig):
        self.cfg = cfg
        self.action_mapping = UniversalAutoregressiveActionMapping(
            config=self.cfg.action_mapping
        )
        self.action_dim = self.action_mapping.get_seq_len()
        logging.warning(
            "!!TESTING ONLY!! DummyDataset is not a real dataset, it just returns random tensors indefinitely."
        )

    def __iter__(self):
        while True:
            frames = torch.randint(
                0,
                255,
                (self.cfg.T, 3, self.cfg.frame_height, self.cfg.frame_width),
                dtype=torch.uint8,
            )
            action_annotations = torch.zeros(
                (self.cfg.T, self.action_dim), dtype=torch.long
            )
            user_action_mask = torch.ones((self.cfg.T), dtype=torch.bool)
            env_subenv_encoding = torch.zeros((1,), dtype=torch.long)
            yield ActionLabelVideoDatasetItem(
                frames=frames,
                action_annotations=action_annotations,
                env_subenv_encoding=env_subenv_encoding,
                user_action_mask=user_action_mask,
            )

    def to_dataloader(self, batch_size: int, prefetch_factor: int = 2):
        return DataLoader(
            self,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            num_workers=28,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            persistent_workers=True,
        )
