import lightning as pl
from elefant.data.action_mapping import UniversalAutoregressiveActionMapping
from elefant.policy_model.config import LightningPolicyConfig
from elefant.im_tokenizer.config import ImageTokenizerConfig
from elefant.policy_model.policy_transformer import (
    PolicyCausalTransformer,
    PolicyCausalTransformerConfig,
    MoEPolicyCausalTransformer,
    MoEPolicyCausalTransformerConfig,
)
from elefant.policy_model.transformer import SparseMoEConfig
import torch
from elefant.torch import eager_assert
from elefant.im_tokenizer import get_tokenizer
from elefant.policy_model.action_decoder import ActionDecoderConfig


class ModelFreePolicy(pl.LightningModule):
    """
    Lighting base module for stage2/stage3 (represents a model-free image -> action policy).
    """

    def __init__(
        self,
        config: LightningPolicyConfig,
        stage_name: str,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.config = config
        self.inference_mode = inference_mode
        self.action_mapping = UniversalAutoregressiveActionMapping(
            config=self.config.shared.action_mapping
        )

        # for stage2 and stage3 policy model, the model attend to real actions
        self.transformer_n_action_tokens = self.action_mapping.get_seq_len()
        self.text_token_size = (
            self.config.shared.text_tokenizer_config.text_embedding_shape[0]
        )
        self.text_tokenizer_embed_dim = (
            self.config.shared.text_tokenizer_config.text_embedding_shape[1]
        )

        self.image_tokenizer = get_tokenizer(
            config.shared.tokenizer,
            config.policy_model.transformer_dim,
            config.shared.frame_height,
            config.shared.frame_width,
        )

        # only used when model_type is sparse_moe
        sparse_moe_config = SparseMoEConfig(
            num_experts=config.policy_model.sparse_moe.num_experts,
            experts_per_token=config.policy_model.sparse_moe.experts_per_token,
            inference_mode=self.inference_mode,
        )
        if config.policy_model.model_type == "sparse_moe":
            raise NotImplementedError("Sparse MoE is not supported anymore.")
            self.bc_transformer = MoEPolicyCausalTransformer(
                config=MoEPolicyCausalTransformerConfig(
                    embed_dim=config.policy_model.transformer_dim,
                    n_steps=config.shared.n_seq_timesteps,
                    n_transformer_layers=config.policy_model.n_transformer_layers,
                    n_q_head=config.policy_model.n_q_head,
                    n_kv_head=config.policy_model.n_kv_head,
                    mask_block_size=config.policy_model.mask_block_size or 128,
                    n_thinking_tokens=config.policy_model.n_thinking_tokens,
                    attention_history_len=config.policy_model.attention_history_len,
                    model_type=config.policy_model.model_type,
                    sparse_moe=sparse_moe_config,
                    action_decoder=ActionDecoderConfig(
                        embed_dim=config.policy_model.action_decoder.embed_dim,
                        n_action_tokens=self.action_mapping.get_seq_len() + 1,
                        input_action_token_dim=config.policy_model.transformer_dim,
                    ),
                ),
                image_tokenizer=self.image_tokenizer,
                inference_mode=inference_mode,
                mask_fn=self._get_transformer_mask_fn(),
            )
        else:
            self.bc_transformer = PolicyCausalTransformer(
                config=PolicyCausalTransformerConfig(
                    embed_dim=config.policy_model.transformer_dim,
                    n_steps=config.shared.n_seq_timesteps,
                    n_transformer_layers=config.policy_model.n_transformer_layers,
                    n_q_head=config.policy_model.n_q_head,
                    n_kv_head=config.policy_model.n_kv_head,
                    mask_block_size=config.policy_model.mask_block_size or 128,
                    n_thinking_tokens=config.policy_model.n_thinking_tokens,
                    attention_history_len=config.policy_model.attention_history_len,
                    model_type=config.policy_model.model_type,
                    action_decoder=ActionDecoderConfig(
                        embed_dim=config.policy_model.action_decoder.embed_dim,
                        n_action_tokens=self.action_mapping.get_seq_len() + 1,
                        input_action_token_dim=config.policy_model.transformer_dim,
                    ),
                    n_kv_sink_tokens=config.policy_model.n_kv_sink_tokens,
                    n_action_tokens=self.transformer_n_action_tokens,
                    text_token_size=self.text_token_size,
                    text_tokenizer_embed_dim=self.text_tokenizer_embed_dim,
                ),
                image_tokenizer=self.image_tokenizer,
                inference_mode=inference_mode,
                mask_fn=self._get_transformer_mask_fn(),
            )

        self.action_seq_len = self.action_mapping.get_seq_len()

    def _get_transformer_mask_fn(self):
        raise NotImplementedError("Subclasses must implement _get_transformer_mask_fn")

    def copy_weights(self, other_model: "ModelFreePolicy"):
        models_to_copy = ["image_tokenizer", "bc_transformer"]
        for model_name in models_to_copy:
            model = getattr(self, model_name)
            other_model_model = getattr(other_model, model_name)
            for my_param, their_param in zip(
                model.parameters(), other_model_model.parameters()
            ):
                my_param.data.copy_(their_param.data)

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        eager_assert(frames.dtype, torch.uint8)
        frames = frames.to(torch.bfloat16)
        frames = frames / 255.0
        return frames
