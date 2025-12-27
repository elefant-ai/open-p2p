import logging

import lightning as pl
import torch
import os
import wandb
import fsspec
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from elefant.data import (
    ActionLabelVideoProtoDataset,
    ActionLabelVideoProtoDatasetConfig,
    UniversalAutoregressiveActionMapping,
    DummyDataset,
    DummyDatasetConfig,
    StructuredAction,
)
from elefant.text_tokenizer.config import TextTokenizerConfig
from elefant.data.rand_augment import BatchRandAugment
from elefant.policy_model.config import LightningPolicyConfig
from elefant.policy_model.model_free import ModelFreePolicy
from lightning.pytorch.callbacks import ModelCheckpoint
from elefant.lightning import AsyncCheckpointIO
from lightning.pytorch.utilities import grad_norm
from typing import List, Optional
from elefant.torch import ELEFANT_WANDB_DIR
from elefant.policy_model.kv_cache import KVCacheState
import pydantic_yaml
from elefant.torch import (
    eager_assert,
    cross_entropy_to_perplexity,
    _sample_from_logits_gpu,
)
from elefant.torch import count_model_parameters
from elefant.data.action_mapping import UniversalAutoregressiveActionMapping
from elefant.metrics import LossMetric
from elefant.policy_model.config import DatasetConfig, ValidationDatasetConfig
from lightning.fabric.utilities.cloud_io import get_filesystem


def upload_model_config(checkpoint_path: str, config):
    """Save the model config to the checkpoint path."""
    # Save the model config to the checkpoint path.
    logging.info(f"Uploading model config to {checkpoint_path}/model_config.yaml")
    with fsspec.open(checkpoint_path + "/model_config.yaml", "wb") as f:
        f.write(pydantic_yaml.to_yaml_str(config).encode())


def upload_action_mapping(
    checkpoint_path: str, action_mapping: UniversalAutoregressiveActionMapping
):
    """Upload the action mapping to the checkpoint path."""
    logging.info(f"Uploading action mapping to {checkpoint_path}/action_mapping.json")
    with fsspec.open(checkpoint_path + "/action_mapping.json", "w") as f:
        f.write(action_mapping.serialize())


def _sample_from_distribution(
    logits: torch.Tensor,
    unif_rand: torch.Tensor,
) -> torch.Tensor:
    eager_assert(unif_rand.ndim, 0)
    probs = torch.softmax(logits, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    cmp = cdf >= unif_rand.unsqueeze(-1)
    return cmp.float().argmax(dim=-1)


class PolicyModelTrainer(ModelFreePolicy):
    def __init__(
        self,
        config: LightningPolicyConfig,
        stage_name: str,
        inference_mode: bool = False,
    ):
        super().__init__(
            config=config, stage_name=stage_name, inference_mode=inference_mode
        )

        self._init_action_mapping()
        self.n_actions = self.action_mapping.get_seq_len()
        self._already_frozen = False
        self._already_unfrozen = False
        self.generate_dummy_text_embed = False
        self.lb_loss_weight = (
            self.config.policy_model.sparse_moe.lb_loss_weight
            if self.config.policy_model.model_type == "sparse_moe"
            else 0
        )
        self.z_loss_weight = self.config.policy_model.z_loss_weight
        if self.config.policy_model.model_type == "sparse_moe":
            self.compile_mode = torch.compile(fullgraph=True)
        else:
            # default compilation mode is without max autotune which gets
            # edited when initializing stage 1/3 with max-autotunes
            self.compile_mode = torch.compile()
        self.rz_loss_weight = (
            self.config.policy_model.sparse_moe.rz_loss_weight
            if self.config.policy_model.model_type == "sparse_moe"
            else 0
        )
        self.num_of_experts = (
            self.config.policy_model.sparse_moe.num_experts
            if self.config.policy_model.model_type == "sparse_moe"
            else 1
        )

        self._init_metrics()
        self.top_p = config.policy_model.top_p

    def setup(self, stage):
        self._init_rand_augment()

    def _init_rand_augment(self):
        ra_cfg = self.config.stage3_finetune.training_dataset.rand_augmentation
        frac = ra_cfg.fraction_augmented
        auglist = ra_cfg.augmentations
        assert frac == 0.0 or (auglist and len(auglist) > 0), (
            "When frac > 0, auglist must be provided and non-empty"
        )
        if frac > 0.0:
            self.rand_augment = BatchRandAugment(augmentations=auglist)
            self.augment_fraction = frac
        else:
            self.rand_augment = None
            self.augment_fraction = 0.0

    # TODO: probably can merge with _action_sampler
    def _keyboard_mouse_action_sampler(
        self,
        action_token: torch.Tensor,
        action_idx: int,
        sampled_actions: StructuredAction,
        sampling_temperature: float = 1.0,
        unif_rand: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = action_token.shape[0]
        eager_assert(
            action_token.shape,
            (T, 1, self.config.policy_model.action_decoder.embed_dim),
        )
        # Sample from the action token and return the sampled action and the action_embedding for the auto-regressive step.
        if action_idx < self.n_keyboard_actions:
            action_logits = self.keyboard_out_logits(action_token)
            eager_assert(action_logits.shape, (T, 1, self.n_keyboard_choices))
        elif action_idx < self.n_keyboard_actions + self.n_mouse_button_actions:
            action_logits = self.mouse_button_out_logits(action_token)
            eager_assert(action_logits.shape, (T, 1, self.n_mouse_button_choices))
        elif action_idx < self.n_keyboard_actions + self.n_mouse_button_actions + 1:
            action_logits = self.mouse_delta_x_out_logits(action_token)
            eager_assert(action_logits.shape, (T, 1, self.n_mouse_x_bins))
        else:
            eager_assert(
                action_idx,
                self.n_keyboard_actions + self.n_mouse_button_actions + 1,
            )
            action_logits = self.mouse_delta_y_out_logits(action_token)
            eager_assert(action_logits.shape, (T, 1, self.n_mouse_y_bins))

        # fast gpu sampling
        action = _sample_from_logits_gpu(
            action_logits,
            sampling_temperature,
            None if unif_rand is None else unif_rand[action_idx],
            top_p=self.top_p,
        ).squeeze(1)

        eager_assert(action.shape, (T, 1))

        # Now embed the action for the auto-regressive step and record the sample action.
        if action_idx < self.n_keyboard_actions:
            sampled_actions.keys[:, action_idx] = action.squeeze(1)
            last_action_in_token = self.key_action_embedding(action)
        elif action_idx < self.n_keyboard_actions + self.n_mouse_button_actions:
            sampled_actions.mouse_buttons[:, action_idx - self.n_keyboard_actions] = (
                action.squeeze(1)
            )
            last_action_in_token = self.mouse_button_embedding(action)
        elif action_idx < self.n_keyboard_actions + self.n_mouse_button_actions + 1:
            sampled_actions.mouse_delta_x[
                :, action_idx - self.n_keyboard_actions - self.n_mouse_button_actions
            ] = action.squeeze(1)
            last_action_in_token = self.mouse_delta_x_embedding(action)
        else:
            eager_assert(
                action_idx,
                self.n_keyboard_actions + self.n_mouse_button_actions + 1,
            )
            sampled_actions.mouse_delta_y[
                :,
                action_idx - self.n_keyboard_actions - self.n_mouse_button_actions - 1,
            ] = action.squeeze(1)
            last_action_in_token = self.mouse_delta_y_embedding(action)

        eager_assert(
            last_action_in_token.shape,
            (T, 1, self.config.policy_model.action_decoder.embed_dim),
        )

        return last_action_in_token

    def on_validation_epoch_start(self):
        # done this way since val_dataloaders are not initialized in _init_ so can't get
        # keys from it(i.e. val_set_names)
        if not self._validation_metrics and self.trainer.val_dataloaders:
            val_set_names = list(self.trainer.val_dataloaders.keys())
            action_types = list(StructuredAction._fields)

            for val_set_name in val_set_names:
                metrics = {
                    "perplexity": LossMetric().to(self.device),
                }
                for action_type in action_types:
                    metric_name = self.action_type_to_metric_name[action_type]
                    metrics[f"perplexity_{metric_name}"] = LossMetric().to(self.device)
                metrics[f"perplexity_lb_loss"] = LossMetric().to(self.device)
                metrics[f"perplexity_rz_loss"] = LossMetric().to(self.device)
                for i in range(self.num_of_experts):
                    metrics[f"expert_{i}_capacity"] = LossMetric().to(self.device)

                self._validation_metrics[val_set_name] = metrics

    def _init_metrics(self):
        action_types = list(StructuredAction._fields)
        self.action_type_to_metric_name = {}
        # mapping from field names to metric names
        for field_name in action_types:
            if field_name == "keys":
                self.action_type_to_metric_name[field_name] = "key"
            elif field_name == "mouse_buttons":
                self.action_type_to_metric_name[field_name] = "mouse_button"
            else:
                # for mouse_delta_x and mouse_delta_y, keep the same name
                self.action_type_to_metric_name[field_name] = field_name

        self._training_loss_metric = LossMetric()
        self._training_ratio_unlabeled_metric = LossMetric()
        self._training_cross_entropy_metric = LossMetric()
        self._training_perplexity_metric = LossMetric()

        for action_type in action_types:
            metric_name = self.action_type_to_metric_name[action_type]
            setattr(self, f"_training_perplexity_{metric_name}_metric", LossMetric())

        setattr(self, f"_training_perplexity_lb_loss_metric", LossMetric())
        setattr(self, f"_training_perplexity_rz_loss_metric", LossMetric())

        for i in range(self.num_of_experts):
            setattr(self, f"_training_expert_{i}_capacity_metric", LossMetric())

        # Initialize empty dict for validation metrics
        self._validation_metrics = {}

    def _init_action_mapping(self):
        # Create the mapping from actions to input tokens and from output tokens to actions.
        self.n_actions = self.action_mapping.get_seq_len()

        self.embedding_std = 0.1

        # Keyboard actions
        self.n_keyboard_actions = self.action_mapping.get_number_of_keyboard_actions()
        self.n_keyboard_choices = self.action_mapping.get_number_of_keyboard_choices()
        self.key_action_embedding = nn.Embedding(
            num_embeddings=self.n_keyboard_choices,
            embedding_dim=self.config.policy_model.action_decoder.embed_dim,
            dtype=torch.bfloat16,
        )
        torch.nn.init.normal_(
            self.key_action_embedding.weight, mean=0.0, std=self.embedding_std
        )

        self.keyboard_out_logits = nn.Linear(
            self.config.policy_model.action_decoder.embed_dim,
            self.action_mapping.get_number_of_keyboard_choices(),
        )

        # Mouse buttons
        self.n_mouse_button_actions = (
            self.action_mapping.get_number_of_mouse_button_actions()
        )
        self.n_mouse_button_choices = (
            self.action_mapping.get_number_of_mouse_button_choices()
        )
        self.mouse_button_embedding = nn.Embedding(
            num_embeddings=self.n_mouse_button_choices,
            embedding_dim=self.config.policy_model.action_decoder.embed_dim,
            dtype=torch.bfloat16,
        )
        torch.nn.init.normal_(
            self.mouse_button_embedding.weight, mean=0.0, std=self.embedding_std
        )

        self.mouse_button_out_logits = nn.Linear(
            self.config.policy_model.action_decoder.embed_dim,
            self.n_mouse_button_choices,
        )

        # Mouse delta x
        self.n_mouse_x_bins = self.action_mapping.get_n_mouse_x_bins()
        self.mouse_delta_x_embedding = nn.Embedding(
            num_embeddings=self.n_mouse_x_bins,
            embedding_dim=self.config.policy_model.action_decoder.embed_dim,
            dtype=torch.bfloat16,
        )
        torch.nn.init.normal_(
            self.mouse_delta_x_embedding.weight, mean=0.0, std=self.embedding_std
        )

        self.mouse_delta_x_out_logits = nn.Linear(
            self.config.policy_model.action_decoder.embed_dim,
            self.n_mouse_x_bins,
        )

        # Mouse delta y
        self.n_mouse_y_bins = self.action_mapping.get_n_mouse_y_bins()
        self.mouse_delta_y_embedding = nn.Embedding(
            num_embeddings=self.n_mouse_y_bins,
            embedding_dim=self.config.policy_model.action_decoder.embed_dim,
            dtype=torch.bfloat16,
        )
        torch.nn.init.normal_(
            self.mouse_delta_y_embedding.weight, mean=0.0, std=self.embedding_std
        )

        self.mouse_delta_y_out_logits = nn.Linear(
            self.config.policy_model.action_decoder.embed_dim,
            self.n_mouse_y_bins,
        )

    def configure_model(self):
        pass

    def online_kv_cache_predict(
        self,
        frame: torch.Tensor,
        idx: torch.Tensor,
        kv_cache_state: List[KVCacheState],
        unif_rand: Optional[torch.Tensor] = None,
        compile: bool = True,
        sampling_temperature: float = 1.0,
        text_tokens_embed: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, List[KVCacheState]]:
        def _action_sampler(
            action_token: torch.Tensor,
            action_idx: int,
            sampled_actions: StructuredAction,
        ) -> torch.Tensor:
            return self._keyboard_mouse_action_sampler(
                action_token,
                action_idx,
                sampled_actions,
                sampling_temperature,
                unif_rand,
            )

        @(torch.compile(fullgraph=True) if compile else lambda f: f)
        def _predict(frame, idx, kv_cache_state, unif_rand, text_tokens_embed):
            frame = self._normalize_frames(frame)
            frame = torch.unsqueeze(frame, 0).unsqueeze(0)

            cache_is_full = False
            if (
                kv_cache_state[0].k_cache.shape[2]
                >= self.bc_transformer.kv_cache.max_seq_len
            ):
                cache_is_full = True

            sampled_action, idx, kv_cache_state = self.bc_transformer.online_forward(
                frame,
                text_tokens_embed=text_tokens_embed,
                idx=idx,
                kv_cache_state=kv_cache_state,
                should_grow_cache=not cache_is_full,
                action_sampler=_action_sampler,
                empty_sampled_action_fn=self.action_mapping.make_empty_action,
                reshape_structured_action_fn=None,
                action_in_to_tokens_fn=self.action_in_to_tokens,
            )

            return sampled_action, idx, kv_cache_state

        with torch.inference_mode():
            return _predict(frame, idx, kv_cache_state, unif_rand, text_tokens_embed)

    # Used for exporting to TensorRT since we can't
    # include the distribution sampling and kv cache (for the moment)
    # in trt.
    def online_full_predict_logits(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        text_tokens_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        frames = self._normalize_frames(frames)
        B, T = frames.shape[0], frames.shape[1]
        action_in = self.action_in_to_tokens(actions)
        action_out, *_ = self.transformer_forward_function(
            frames, action_in, text_tokens_embed
        )
        action_logits = self.action_out_tokens_to_logits(action_out)
        eager_assert(
            action_logits.keys.shape,
            (
                B,
                T,
                self.n_keyboard_actions,
                self.n_keyboard_choices,
            ),
        )
        return action_logits

    def online_full_predict(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        kv_cache_state: List[KVCacheState] = None,
        sampling_temperature: float = 1.0,
        text_tokens_embed: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        @(torch.compile(fullgraph=True) if compile else lambda f: f)
        def _predict(frames, actions, kv_cache_state, text_tokens_embed):
            T = frames.shape[1]
            action_logits = self.online_full_predict_logits(
                frames, actions, text_tokens_embed
            )

            key_idx = _sample_from_logits_gpu(
                action_logits.keys,
                sampling_temperature,
                None,
                top_p=self.top_p,
            ).squeeze(-1)

            mouse_button_idx = _sample_from_logits_gpu(
                action_logits.mouse_buttons,
                sampling_temperature,
                None,
                top_p=self.top_p,
            ).squeeze(-1)

            mouse_delta_x_idx = _sample_from_logits_gpu(
                action_logits.mouse_delta_x,
                sampling_temperature,
                None,
                top_p=self.top_p,
            ).squeeze(-1)

            mouse_delta_y_idx = _sample_from_logits_gpu(
                action_logits.mouse_delta_y,
                sampling_temperature,
                None,
                top_p=self.top_p,
            ).squeeze(-1)

            action_out = StructuredAction(
                keys=key_idx,
                mouse_buttons=mouse_button_idx,
                mouse_delta_x=mouse_delta_x_idx,
                mouse_delta_y=mouse_delta_y_idx,
            )

            eager_assert(action_out.keys.shape, (1, T, self.n_keyboard_actions))
            return action_out, action_logits

        with torch.inference_mode():
            return _predict(frames, actions, kv_cache_state, text_tokens_embed)

    def action_in_to_tokens(
        self, action_in: StructuredAction, idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        If idx is provided, this will return dummy actions that match the shape of the frame input.
        If idx is not provided, this will return the actual actions embeddings.
        """
        B, T, _ = action_in.keys.shape
        eager_assert(action_in.keys.shape, (B, T, self.n_keyboard_actions))
        eager_assert(action_in.mouse_buttons.shape, (B, T, self.n_mouse_button_actions))
        eager_assert(action_in.mouse_delta_x.shape, (B, T, 1))
        eager_assert(action_in.mouse_delta_y.shape, (B, T, 1))

        if idx is None:
            idx = torch.arange(B)
        key_action_embedding = self.key_action_embedding(action_in.keys[idx])
        mouse_button_embedding = self.mouse_button_embedding(
            action_in.mouse_buttons[idx]
        )
        mouse_delta_x_embedding = self.mouse_delta_x_embedding(
            action_in.mouse_delta_x[idx]
        )
        mouse_delta_y_embedding = self.mouse_delta_y_embedding(
            action_in.mouse_delta_y[idx]
        )
        action_embedding = torch.cat(
            [
                key_action_embedding,
                mouse_button_embedding,
                mouse_delta_x_embedding,
                mouse_delta_y_embedding,
            ],
            dim=2,
        )
        eager_assert(
            action_embedding.shape,
            (
                B if idx is None else len(idx),
                T,
                self.n_actions,
                self.config.policy_model.action_decoder.embed_dim,
            ),
        )
        return action_embedding

    def action_out_tokens_to_logits(
        self, action_out_tokens: torch.Tensor
    ) -> torch.Tensor:
        # Any changes here should be reflected in the online_kv_cache_predict function.

        B, T, N, D = action_out_tokens.shape
        eager_assert(N, self.n_actions)

        key_logits = self.keyboard_out_logits(
            action_out_tokens[:, :, : self.n_keyboard_actions, :]
        )
        eager_assert(
            key_logits.shape,
            (
                B,
                T,
                self.n_keyboard_actions,
                self.action_mapping.get_number_of_keyboard_choices(),
            ),
        )
        mouse_button_logits = self.mouse_button_out_logits(
            action_out_tokens[
                :,
                :,
                self.n_keyboard_actions : self.n_keyboard_actions
                + self.n_mouse_button_actions,
                :,
            ]
        )
        eager_assert(
            mouse_button_logits.shape,
            (
                B,
                T,
                self.n_mouse_button_actions,
                self.action_mapping.get_number_of_mouse_button_choices(),
            ),
        )
        mouse_delta_x_logits = self.mouse_delta_x_out_logits(
            action_out_tokens[
                :,
                :,
                self.n_keyboard_actions
                + self.n_mouse_button_actions : self.n_keyboard_actions
                + self.n_mouse_button_actions
                + 1,
                :,
            ]
        )
        eager_assert(
            mouse_delta_x_logits.shape,
            (B, T, 1, self.action_mapping.get_n_mouse_x_bins()),
        )
        mouse_delta_y_logits = self.mouse_delta_y_out_logits(
            action_out_tokens[
                :,
                :,
                self.n_keyboard_actions
                + self.n_mouse_button_actions
                + 1 : self.n_keyboard_actions + self.n_mouse_button_actions + 2,
                :,
            ]
        )
        eager_assert(
            mouse_delta_y_logits.shape,
            (B, T, 1, self.action_mapping.get_n_mouse_y_bins()),
        )

        return StructuredAction(
            keys=key_logits,
            mouse_buttons=mouse_button_logits,
            mouse_delta_x=mouse_delta_x_logits,
            mouse_delta_y=mouse_delta_y_logits,
        )

    def on_before_optimizer_step(self, optimizer):
        # inspect (unscaled) gradients here
        if self.global_step % 100 == 0:
            self.log_dict(grad_norm(self, norm_type=2))

    def init_from_stage2_model(self, stage2_model):
        super().copy_weights(stage2_model)

    def _calculate_z_loss(self, action_logits, masked_labels):
        key_z_loss = (
            (action_logits.keys.view(-1, self.n_keyboard_choices).logsumexp(-1).pow(2))
            * (masked_labels.keys.view(-1) != -100)
        ).mean()
        mouse_button_z_loss = (
            (
                action_logits.mouse_buttons.view(-1, self.n_mouse_button_choices)
                .logsumexp(-1)
                .pow(2)
            )
            * (masked_labels.mouse_buttons.view(-1) != -100)
        ).mean()
        mouse_delta_x_z_loss = (
            (
                action_logits.mouse_delta_x.view(-1, self.n_mouse_x_bins)
                .logsumexp(-1)
                .pow(2)
            )
            * (masked_labels.mouse_delta_x.view(-1) != -100)
        ).mean()
        mouse_delta_y_z_loss = (
            (
                action_logits.mouse_delta_y.view(-1, self.n_mouse_y_bins)
                .logsumexp(-1)
                .pow(2)
            )
            * (masked_labels.mouse_delta_y.view(-1) != -100)
        ).mean()
        return (
            key_z_loss,
            mouse_button_z_loss,
            mouse_delta_x_z_loss,
            mouse_delta_y_z_loss,
        )

    def _calculate_loss(self, batch, actions_in, masked_labels, text_tokens_embed):
        """
        Calculate the loss for the given batch of actions.
        batch: batch from dataloader
        actions_in: action sequence correspond to frames, use ground truth action for labeled data and pseudo labels for unlabeled data
        masked_labels: action sequence masked with user action mask, same as action_in for unlabeled data
        """
        frames = self._normalize_frames(batch.frames)
        batch_size = batch.frames.shape[0]
        T = batch.frames.shape[1]
        action_embeddings_in = self.action_in_to_tokens(actions_in)
        eager_assert(
            action_embeddings_in.shape,
            (
                batch_size,
                T,
                self.n_actions,
                self.config.policy_model.action_decoder.embed_dim,
            ),
        )
        action_out_embeddings, _, auxiliary_losses, auxiliary_outputs = (
            self.transformer_forward_function(
                frames, action_embeddings_in, text_tokens_embed
            )
        )
        eager_assert(
            action_out_embeddings.shape,
            (
                batch_size,
                T,
                self.n_actions,
                self.config.policy_model.action_decoder.embed_dim,
            ),
        )

        action_logits = self.action_out_tokens_to_logits(action_out_embeddings)

        eager_assert(
            action_logits.keys.shape,
            (batch_size, T, self.n_keyboard_actions, self.n_keyboard_choices),
        )
        eager_assert(
            action_logits.mouse_buttons.shape,
            (batch_size, T, self.n_mouse_button_actions, self.n_mouse_button_choices),
        )
        eager_assert(
            action_logits.mouse_delta_x.shape,
            (batch_size, T, 1, self.n_mouse_x_bins),
        )
        eager_assert(
            action_logits.mouse_delta_y.shape,
            (batch_size, T, 1, self.n_mouse_y_bins),
        )

        key_loss = F.cross_entropy(
            input=action_logits.keys.view(-1, self.n_keyboard_choices),
            target=masked_labels.keys.view(-1),
            ignore_index=-100,
        )
        mouse_button_loss = F.cross_entropy(
            input=action_logits.mouse_buttons.view(-1, self.n_mouse_button_choices),
            target=masked_labels.mouse_buttons.view(-1),
            ignore_index=-100,
        )
        mouse_delta_x_loss = F.cross_entropy(
            input=action_logits.mouse_delta_x.view(-1, self.n_mouse_x_bins),
            target=masked_labels.mouse_delta_x.view(-1),
            ignore_index=-100,
        )
        mouse_delta_y_loss = F.cross_entropy(
            input=action_logits.mouse_delta_y.view(-1, self.n_mouse_y_bins),
            target=masked_labels.mouse_delta_y.view(-1),
            ignore_index=-100,
        )
        lb_loss = auxiliary_losses.get("lb_loss", torch.tensor(0.0))
        rz_loss = auxiliary_losses.get("rz_loss", torch.tensor(0.0))
        losses = {
            "key": key_loss,
            "mouse_button": mouse_button_loss,
            "mouse_delta_x": mouse_delta_x_loss,
            "mouse_delta_y": mouse_delta_y_loss,
            "lb_loss": lb_loss,
            "rz_loss": rz_loss,
        }
        key_z_loss, mouse_button_z_loss, mouse_delta_x_z_loss, mouse_delta_y_z_loss = (
            self._calculate_z_loss(action_logits, masked_labels)
        )
        loss = (
            (key_loss) / torch.log(torch.tensor(self.n_keyboard_choices))
            + (mouse_button_loss) / torch.log(torch.tensor(self.n_mouse_button_choices))
            + (mouse_delta_x_loss) / torch.log(torch.tensor(self.n_mouse_x_bins))
            + (mouse_delta_y_loss) / torch.log(torch.tensor(self.n_mouse_y_bins))
            + (
                key_z_loss
                + mouse_button_z_loss
                + mouse_delta_x_z_loss
                + mouse_delta_y_z_loss
            )
            * self.z_loss_weight
            + lb_loss * self.lb_loss_weight
            + rz_loss * self.rz_loss_weight
        )
        cross_entropy_loss = (
            key_loss / torch.log(torch.tensor(self.n_keyboard_choices))
            + mouse_button_loss / torch.log(torch.tensor(self.n_mouse_button_choices))
            + mouse_delta_x_loss / torch.log(torch.tensor(self.n_mouse_x_bins))
            + mouse_delta_y_loss / torch.log(torch.tensor(self.n_mouse_y_bins))
        )
        return loss, cross_entropy_loss, losses, auxiliary_outputs

    def _create_target_and_masked_labels(self, batch):
        batch_size = batch.frames.shape[0]
        T = batch.frames.shape[1]
        user_action_mask = batch.user_action_mask
        system_action_mask = batch.system_action_mask
        valid_frame_mask = batch.valid_frame_mask
        eager_assert(user_action_mask.shape, (batch_size, T))
        eager_assert(valid_frame_mask.shape, (batch_size, T))
        eager_assert(system_action_mask.shape, (batch_size, T))

        # If not using IDM
        # only compute loss on frames that are both user-labeled and valid
        effective_mask = self._compute_effective_mask(
            user_action_mask, valid_frame_mask, system_action_mask
        )
        # effective_mask = user_action_mask & valid_frame_mask
        actions_in = batch.action_annotations
        masked_labels = StructuredAction(
            keys=torch.where(
                effective_mask.unsqueeze(2),
                batch.action_annotations.keys,
                -100,
            ),
            mouse_buttons=torch.where(
                effective_mask.unsqueeze(2),
                batch.action_annotations.mouse_buttons,
                -100,
            ),
            mouse_delta_x=torch.where(
                effective_mask.unsqueeze(2),
                batch.action_annotations.mouse_delta_x,
                -100,
            ),
            mouse_delta_y=torch.where(
                effective_mask.unsqueeze(2),
                batch.action_annotations.mouse_delta_y,
                -100,
            ),
        )
        ratio_unlabeled = torch.zeros(
            (), dtype=torch.float32, device=user_action_mask.device
        )
        return actions_in, masked_labels, ratio_unlabeled

    def _apply_augmentations(self, batch):
        """Apply random augmentations to frames on GPU"""
        if self.rand_augment is None or self.augment_fraction == 0.0:
            return batch

        should_augment = torch.rand(1) < self.augment_fraction

        def _augment_fn(frames):
            frames = self.rand_augment(frames)
            return frames

        def _no_augment_fn(frames):
            return frames

        frames = batch.frames
        if should_augment:
            frames = _augment_fn(frames)

        # TODO: would be nice to have this compiled and use torch.cond
        # frames = torch.cond(should_augment, _augment_fn, _no_augment_fn, (frames,))
        return batch._replace(frames=frames)

    def training_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
            logging.info(
                f"First training step starting (compilation may take awhile). rank={self.trainer.global_rank}"
            )

        batch = self._apply_augmentations(batch)
        text_tokens_embed = batch.text_embeddings

        @self.compile_mode
        def compiled_training_step(batch):
            with torch.no_grad():
                actions_in, masked_labels, ratio_unlabeled = (
                    self._create_target_and_masked_labels(batch)
                )
            # The actual optimization happens here.
            loss, cross_entropy_loss, losses, auxiliary_outputs = self._calculate_loss(
                batch, actions_in, masked_labels, text_tokens_embed
            )

            auxiliary_outputs["ratio_unlabeled"] = ratio_unlabeled
            return loss, losses, cross_entropy_loss, auxiliary_outputs

        loss, losses, cross_entropy_loss, auxiliary_outputs = compiled_training_step(
            batch
        )

        if self.trainer.global_step == 0:
            logging.info(
                f"First training step completed. rank={self.trainer.global_rank}"
            )

        self._training_loss_metric.update(loss)
        self._training_ratio_unlabeled_metric.update(
            auxiliary_outputs["ratio_unlabeled"]
        )
        self._training_cross_entropy_metric.update(cross_entropy_loss)
        self._training_perplexity_metric.update(cross_entropy_to_perplexity(loss))

        for k, v in losses.items():
            getattr(self, f"_training_perplexity_{k}_metric").update(
                cross_entropy_to_perplexity(v)
            )
        for i in range(self.num_of_experts):
            metric = getattr(self, f"_training_expert_{i}_capacity_metric")
            metric.update(auxiliary_outputs["num_tokens_per_expert"][i])

        # Only log on the final gradient accumulation step.
        # if not self.trainer.fit_loop._should_accumulate():
        # Accumulation gets weird at the end of the epoch.
        # if self.trainer.fit_loop.epoch_loop._accumulated_batches_reached():
        # TODO: proper fix, right not gradient accumulation does not work with DDP.
        if self.trainer.global_step % 50 == 0:
            self.log(
                "training_loss",
                self._training_loss_metric.compute(),
                sync_dist=True,
                add_dataloader_idx=False,
                on_step=True,
            )
            self.log(
                "training_cross_entropy",
                self._training_cross_entropy_metric.compute(),
                sync_dist=True,
                add_dataloader_idx=False,
                on_step=True,
            )
            self.log(
                "training_perplexity",
                self._training_perplexity_metric.compute(),
                sync_dist=True,
                add_dataloader_idx=False,
                on_step=True,
            )
            self.log(
                "training_ratio_unlabeled",
                self._training_ratio_unlabeled_metric.compute(),
                sync_dist=True,
                add_dataloader_idx=False,
                on_step=True,
            )
            for k in losses.keys():
                metric = getattr(self, f"_training_perplexity_{k}_metric")
                self.log(
                    f"training_perplexity_{k}",
                    metric.compute(),
                    sync_dist=True,
                    add_dataloader_idx=False,
                    on_step=True,
                )
                metric.reset()
            for i in range(self.num_of_experts):
                metric = getattr(self, f"_training_expert_{i}_capacity_metric")
                self.log(
                    f"training_expert_{i}_capacity",
                    metric.compute(),
                    sync_dist=True,
                    add_dataloader_idx=False,
                    on_step=True,
                )
                metric.reset()
            self._training_loss_metric.reset()
            self._training_cross_entropy_metric.reset()
            self._training_perplexity_metric.reset()
            self._training_ratio_unlabeled_metric.reset()

        # Record the total number of frames seen in training.
        # This depends on the number of optimizer steps (global_step), number of devices, and batch size per device.
        B, T = batch.frames.shape[0], batch.frames.shape[1]
        n_global_training_frames = (
            self.trainer.global_step
            * self.trainer.accumulate_grad_batches
            * self.trainer.num_devices
            * T
            * B
        )
        self.log(
            "n_global_training_frames",
            n_global_training_frames,
            add_dataloader_idx=False,
            on_step=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        text_tokens_embed = batch.text_embeddings

        @self.compile_mode
        def _compiled_validation_step(batch, text_tokens_embed):
            actions_in, masked_labels, _ = self._create_target_and_masked_labels(batch)
            loss, _, losses, auxiliary_outputs = self._calculate_loss(
                batch, actions_in, masked_labels, text_tokens_embed
            )
            return loss, losses, auxiliary_outputs

        loss, losses, auxiliary_outputs = _compiled_validation_step(
            batch, text_tokens_embed
        )
        val_set_name = list(self.trainer.val_dataloaders.keys())[dataloader_idx]
        val_metrics = self._validation_metrics[val_set_name]
        val_metrics["perplexity"].update(cross_entropy_to_perplexity(loss))
        val_metrics["perplexity_rz_loss"].update(
            cross_entropy_to_perplexity(losses["rz_loss"])
        )
        val_metrics["perplexity_lb_loss"].update(
            cross_entropy_to_perplexity(losses["lb_loss"])
        )
        for k, v in losses.items():
            val_metrics[f"perplexity_{k}"].update(cross_entropy_to_perplexity(v))
        for i in range(self.num_of_experts):
            val_metrics[f"expert_{i}_capacity"].update(
                auxiliary_outputs["num_tokens_per_expert"][i]
            )

    def on_validation_epoch_end(self):
        """Compute and log all validation metrics at the end of validation epoch"""
        for val_set_name, metrics in self._validation_metrics.items():
            # Compute, log, and reset all metrics for this validation set
            for metric_name, metric in metrics.items():
                self.log(
                    f"{val_set_name}_validation_{metric_name}",
                    metric.compute(),
                    sync_dist=True,
                    add_dataloader_idx=False,
                    on_step=False,
                    on_epoch=True,
                )
                metric.reset()

    def configure_optimizers(self):
        raise NotImplementedError("Not implemented for base class.")

    def _get_text_embedding_dim(self):
        return self.config.shared.text_tokenizer_config.text_embedding_shape[-1]

    def _get_text_tokenizer_name(self):
        return self.config.shared.text_tokenizer_config.text_tokenizer_name


class Stage3LabelledBCLightning(PolicyModelTrainer):
    def __init__(
        self,
        config: LightningPolicyConfig,
        inference_mode: bool = False,
    ):
        super().__init__(
            config,
            stage_name="stage3_finetune",
            inference_mode=inference_mode,
        )
        if self.config.policy_model.model_type == "sparse_moe":
            self.compile_mode = torch.compile(fullgraph=True)
        else:
            self.compile_mode = torch.compile(fullgraph=True, mode="max-autotune")

        self.transformer_forward_function = self.bc_transformer.forward

    def _get_transformer_mask_fn(self):
        # Use the default, causal mask.
        return None

    def configure_optimizers(self):
        assert not self.inference_mode
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.stage3_finetune.optim.learning_rate,
            betas=(
                self.config.stage3_finetune.optim.beta_1,
                self.config.stage3_finetune.optim.beta_2,
            ),
            weight_decay=self.config.stage3_finetune.optim.weight_decay,
            fused=True,
        )
        return [optimizer]

    def on_train_batch_start(self, batch, batch_idx):
        global_step = self.trainer.global_step
        # This will get called multiple times if gradient accumulation is used.
        if (
            global_step == 0
            and self.config.stage3_finetune.freeze_transformer_layers_for_steps > 0
            and not self._already_frozen
        ):
            logging.warning("Freezing transformer layers.")
            for param in self.bc_transformer.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.parameters():
                param.requires_grad = False
            self._already_frozen = True
        elif (
            global_step
            == self.config.stage3_finetune.freeze_transformer_layers_for_steps
            and not self._already_unfrozen
        ):
            logging.warning("Unfreezing transformer layers.")
            for param in self.bc_transformer.parameters():
                param.requires_grad = True
            for param in self.image_tokenizer.parameters():
                param.requires_grad = True
            self._already_unfrozen = True

    def _compute_effective_mask(
        self, user_action_mask, valid_frame_mask, system_action_mask
    ):
        return valid_frame_mask & (user_action_mask)


def _init_stage3_model(config: LightningPolicyConfig) -> Stage3LabelledBCLightning:
    logging.warning("Initializing stage3 model with random weights.")
    assert config.stage3_finetune.init.stage2_model_path is None, (
        "stage2_model_path is not allowed when initializing with random weights"
    )
    model = Stage3LabelledBCLightning(config)

    return model


class SupervisedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: LightningPolicyConfig,
        training_dataset_cfg: DatasetConfig,
        validation_dataset_cfgs: List[ValidationDatasetConfig],
        stage_name: str,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.training_dataset_cfg = training_dataset_cfg
        self.validation_dataset_cfgs = validation_dataset_cfgs
        self.stage_name = stage_name
        self._setup_completed = False

        assert self.cfg.stage3_finetune.accumulate_grad_batches == 1, (
            "accumulate_grad_batches can deadlock with multiple GPUs"
        )
        self.text_tokenizer_config = TextTokenizerConfig(
            text_tokenizer_name=self.cfg.shared.text_tokenizer_config.text_tokenizer_name,
            text_embedding_shape=self.cfg.shared.text_tokenizer_config.text_embedding_shape,
            text_annotation_model_version=self.cfg.shared.text_tokenizer_config.text_annotation_model_version,
        )

    def _init_train_dataset(self):
        return ActionLabelVideoProtoDataset(
            ActionLabelVideoProtoDatasetConfig(
                frame_height=self.cfg.shared.frame_height,
                frame_width=self.cfg.shared.frame_width,
                local_prefix=self.training_dataset_cfg.local_prefix,
                shuffle=True,
                T=self.cfg.shared.n_seq_timesteps,
                shuffle_buffer_size=self.training_dataset_cfg.shuffle_buffer_size_per_gpu
                * self.world_size,
                n_preprocess_workers_per_iter_worker=self.training_dataset_cfg.n_preprocess_threads_per_gpu,
                preprocessed_chunks_queue_size=self.training_dataset_cfg.preprocessed_chunks_queue_size_per_gpu,
                warn_on_starvation=self.training_dataset_cfg.warn_on_starvation,
                action_mapping=self.cfg.shared.action_mapping,
                always_labelled=self.training_dataset_cfg.always_labelled,
                rand_augmentation=self.training_dataset_cfg.rand_augmentation,
                drop_chunks_with_only_system_actions=self._should_drop_chunks_with_only_system_actions(),
                batch_size=self.training_dataset_cfg.batch_size,
                shuffled_chunks_queue_size=self.training_dataset_cfg.shuffled_chunks_queue_size_per_gpu,
                dataset_worker_prefetch_factor=self.training_dataset_cfg.dataset_worker_prefetch_factor,
                # We don't need to multiple this by num_gpus because dataset will be run multiple times.
                dataset_worker_num_workers=self.training_dataset_cfg.dataset_worker_num_workers_per_gpu,
                dataset_unique_id="training_dataset",
                text_tokenizer_config=self.text_tokenizer_config,
            ),
            device="cpu",
        )

    def _init_dummy_dataset(self):
        return DummyDataset(
            DummyDatasetConfig(
                frame_height=self.cfg.shared.frame_height,
                frame_width=self.cfg.shared.frame_width,
                T=self.cfg.shared.n_seq_timesteps,
                action_mapping=self.cfg.stage3_finetune.action_mapping,
            ),
        )

    def setup(self, stage: str):
        try:
            self.global_rank = self.trainer.global_rank
            self.world_size = self.trainer.world_size
            logging.info(
                f"Setting up datasets. global_rank: {self.global_rank}, world_size: {self.world_size}, stage {stage}"
            )
            if self._setup_completed:
                logging.info(
                    f"Setup already completed. global_rank: {self.global_rank}, world_size: {self.world_size}, stage {stage}"
                )
                self.train_dataset._dataset_worker_generation += 1
                logging.info(
                    f"Train dataset worker generation: {self.train_dataset._dataset_worker_generation}, rank: {self.global_rank}"
                )
                for k, v in self.validation_datasets.items():
                    v._dataset_worker_generation += 1
                    logging.info(
                        f"Validation dataset {k} worker generation: {v._dataset_worker_generation}, rank: {self.global_rank}"
                    )
                return
            self._setup_completed = True

            # You can use the dummy dataset for testing speed.
            # self.train_dataset = self._init_dummy_dataset()
            self.train_dataset = self._init_train_dataset()
            self._train_dataloader = self.train_dataset.to_dataloader()
        except Exception as e:
            logging.warning(
                f"this warning should only happen during offline validaiton."
            )
            self.world_size = 1

        self.validation_datasets = {}
        for i, validation_dataset_cfg in enumerate(self.validation_dataset_cfgs):
            validation_dataset = ActionLabelVideoProtoDataset(
                ActionLabelVideoProtoDatasetConfig(
                    frame_height=self.cfg.shared.frame_height,
                    frame_width=self.cfg.shared.frame_width,
                    local_prefix=validation_dataset_cfg.local_prefix,
                    shuffle=False,
                    T=self.cfg.shared.n_seq_timesteps,
                    shuffle_buffer_size=validation_dataset_cfg.shuffle_buffer_size_per_gpu
                    * self.world_size,
                    n_preprocess_workers_per_iter_worker=validation_dataset_cfg.n_preprocess_threads_per_gpu,
                    preprocessed_chunks_queue_size=validation_dataset_cfg.preprocessed_chunks_queue_size_per_gpu,
                    # For validation we always use only human data.
                    drop_chunks_with_only_system_actions=self._should_drop_chunks_with_only_system_actions(),
                    warn_on_starvation=validation_dataset_cfg.warn_on_starvation,
                    action_mapping=self.cfg.shared.action_mapping,
                    always_labelled=validation_dataset_cfg.always_labelled,
                    rand_augmentation=validation_dataset_cfg.rand_augmentation,
                    ignore_iterator_reset=True,
                    batch_size=validation_dataset_cfg.batch_size,
                    dataset_worker_prefetch_factor=validation_dataset_cfg.dataset_worker_prefetch_factor,
                    dataset_worker_num_workers=validation_dataset_cfg.dataset_worker_num_workers_per_gpu,
                    shuffled_chunks_queue_size=validation_dataset_cfg.shuffled_chunks_queue_size_per_gpu,
                    dataset_unique_id=f"{validation_dataset_cfg.validation_name}_{i}",
                    text_tokenizer_config=self.text_tokenizer_config,
                ),
                device="cpu",
            )
            self.validation_datasets[validation_dataset_cfg.validation_name] = (
                validation_dataset
            )

        self._val_dataloaders = {
            k: d.to_dataloader() for k, d in self.validation_datasets.items()
        }

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloaders

    def get_action_mapping(self):
        return UniversalAutoregressiveActionMapping(
            config=self.cfg.shared.action_mapping
        )


class Stage3DataModule(SupervisedDataModule):
    def __init__(self, cfg: LightningPolicyConfig):
        super().__init__(
            cfg=cfg,
            training_dataset_cfg=cfg.stage3_finetune.training_dataset,
            validation_dataset_cfgs=cfg.stage3_finetune.validation_datasets,
            stage_name="stage3_finetune",
        )

    def _should_drop_chunks_with_only_system_actions(self):
        return True


def train_stage3_finetune(config: LightningPolicyConfig):
    datamodule = Stage3DataModule(config)
    # This is for start_experiment.py type jobs
    run_id = getattr(config.wandb, "run_id", None) or os.environ.get("WANDB_RUN_ID")
    if not run_id:
        run_id = wandb.util.generate_id()

    os.environ["WANDB_RUN_ID"] = run_id
    config.wandb.run_id = run_id

    wandb_logger = pl.pytorch.loggers.WandbLogger(
        entity="elefantai",
        project=config.wandb.project,
        name=config.wandb.exp_name + "_stage3_finetune",
        version=run_id,
        id=run_id,
        log_model=False,
        save_code=False,
        save_dir=ELEFANT_WANDB_DIR,
        config=config.model_dump(),
        group=config.wandb.exp_name,
        job_type="train",
        mode="online" if config.wandb.enabled else "disabled",
    )

    checkpoint_path = f"{config.shared.output_path}/stage3_finetune"
    upload_model_config(checkpoint_path, config)
    upload_action_mapping(checkpoint_path, datamodule.get_action_mapping())

    async_checkpointer = AsyncCheckpointIO()
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        every_n_train_steps=config.stage3_finetune.save_every_n_steps,
        filename="checkpoint-{step:08d}",
        enable_version_counter=False,
        save_top_k=-1,
    )

    if torch.cuda.device_count() > 1:
        logging.info(f"Using DDP strategy with {torch.cuda.device_count()} GPUs")
        strategy = pl.pytorch.strategies.DDPStrategy(find_unused_parameters=True)
    else:
        logging.info("Using SingleDeviceStrategy for single GPU.")
        # strategy = pl.pytorch.strategies.SingleDeviceStrategy(accelerator="auto")
        # Setting strategy with single GPU explicitly seems to error.
        # https://github.com/Lightning-AI/pytorch-lightning/issues/18902
        strategy = "auto"

    trainer = pl.Trainer(
        plugins=[async_checkpointer],
        callbacks=[checkpoint_callback],
        accelerator="auto",
        # For debugging it can be useful to set devices to 1.
        # for simpler stack traces etc.
        devices="auto",
        max_steps=config.stage3_finetune.n_training_steps,
        logger=wandb_logger,
        # We multiply by accumulate_grad_batches to get the number of steps between validation steps in "real" steps.
        val_check_interval=config.stage3_finetune.validation_step_interval
        * config.stage3_finetune.accumulate_grad_batches,
        limit_val_batches=config.stage3_finetune.n_validation_steps,
        check_val_every_n_epoch=None,
        precision=config.shared.precision,
        accumulate_grad_batches=config.stage3_finetune.accumulate_grad_batches,
        fast_dev_run=config.shared.fast_dev_run,
        strategy=strategy,
        # We already run validation before training starts.
        num_sanity_val_steps=0,
        # profiler="simple",
    )

    # Initialize model on the correct device using PyTorch Lightning's device management
    # Disable if using FSDP or DeepSpeed.
    # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
    with trainer.init_module():
        model = _init_stage3_model(config)

    total_params, expert_params = count_model_parameters(model)
    logging.info(
        f"Total parameters: {total_params}, Expert parameters: {expert_params}"
    )

    trainer.fit(model, datamodule)

    wandb_logger.experiment.finish()
    return async_checkpointer.get_final_checkpoint()
