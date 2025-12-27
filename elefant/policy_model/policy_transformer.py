from typing import List, Optional, Callable
from elefant.policy_model.transformer import TransformerConfig, MoETransformer
from elefant.policy_model.action_decoder import ActionDecoderConfig
from elefant.im_tokenizer import ImageBaseTokenizer
import torch
import torch.utils._pytree as pytree
from elefant.policy_model.transformer import (
    Transformer,
    TransformerSelfAttentionLayer,
    SparseMoEConfig,
)
from elefant.policy_model.kv_cache import RollingStepKVCache, KVCacheState
from elefant.torch import eager_assert
import torch.nn as nn
from elefant.policy_model.action_decoder import ActionDecoder
from torch.nn.attention import flex_attention as fa
from typing import Tuple


class PolicyCausalTransformerConfig(TransformerConfig):
    n_steps: int = 3
    n_thinking_tokens: int = 1
    # This needs to be set to the number of actions in the action mapping + 1
    n_action_tokens: int = 5
    mask_block_size: int = 128
    attention_history_len: List[int]
    n_kv_sink_tokens: int = 1

    action_decoder: ActionDecoderConfig
    text_token_size: int
    text_tokenizer_embed_dim: int


class MoEPolicyCausalTransformerConfig(PolicyCausalTransformerConfig):
    sparse_moe: SparseMoEConfig = SparseMoEConfig()


def _img_policy_causal_mask(
    layer_idx: int,
    n_img_tokens: int,
    n_thinking_tokens: int,
    n_action_tokens: int,
    history_len: Optional[int] = None,
    n_text_tokens: int = 0,
    n_kv_sink_tokens: int = 0,
):
    """
    This is a causal mask for the image policy transformer.
    Any image or thinking token can attend to any other image or thinking token but in order to causal with respect
    to actions the action tokens must be causally auto-regressive, also prior actions are masked out to encourage the
    model to learn causally.
    There are extra KV sink tokens at the start of the sequence, always attendable by query
    """

    def _mask(b, h, q_idx, kv_idx):
        is_sink = kv_idx < n_kv_sink_tokens

        # Shift non-sink KV positions into the base index space [0, max_seq_len)
        kv_idx_base = kv_idx - n_kv_sink_tokens

        one_step_len = (
            n_img_tokens + n_text_tokens + n_thinking_tokens + n_action_tokens
        )
        # However, within a single image it's fine to look ahead.
        token_to_action_out = n_img_tokens + n_thinking_tokens + n_text_tokens

        def _step_and_img_mask(idx):
            is_img_or_text_or_thinking_token = (idx % one_step_len) < (
                token_to_action_out
            )
            step_idx = idx // one_step_len

            return is_img_or_text_or_thinking_token, step_idx

        def _step_and_real_action_mask(idx):
            is_action_token = (idx % one_step_len) > (token_to_action_out)
            step_idx = idx // one_step_len

            return is_action_token, step_idx

        def _step_and_action_out_mask(idx):
            is_action_out_token = (idx % one_step_len) == (token_to_action_out)
            step_idx = idx // one_step_len

            return is_action_out_token, step_idx

        q_is_img, q_step_idx = _step_and_img_mask(q_idx)
        kv_is_img, kv_step_idx = _step_and_img_mask(kv_idx_base)
        q_is_real_action, _ = _step_and_real_action_mask(q_idx)
        kv_is_real_action, _ = _step_and_real_action_mask(kv_idx_base)
        q_is_action_out, _ = _step_and_action_out_mask(q_idx)
        kv_is_action_out, _ = _step_and_action_out_mask(kv_idx_base)
        full_mask = (
            (
                q_is_img & ~kv_is_action_out & (kv_step_idx < q_step_idx)
            )  # image attend to anything except for action out
            | (
                q_is_action_out & ~kv_is_action_out & (kv_step_idx < q_step_idx)
            )  # action out attend to anything except for action out
            | (
                q_is_real_action & ~kv_is_action_out & (kv_step_idx < q_step_idx)
            )  # real action attend to anything except for action out
            | (q_is_img & kv_is_img & (kv_step_idx == q_step_idx))  # img attend to img
            | (
                q_is_action_out
                & (kv_is_img | kv_is_action_out)
                & (kv_step_idx == q_step_idx)
            )  # action out attend to action out or img
            | (
                q_is_real_action & ~kv_is_action_out & (kv_step_idx == q_step_idx)
            )  # real action attend to anything except for action out
        )

        if history_len is not None:
            history_mask = (q_step_idx - kv_step_idx) <= history_len
            full_mask = full_mask & history_mask
        return is_sink | full_mask

    return _mask


class PolicyCausalTransformer(torch.nn.Module):
    def __init__(
        self,
        config: PolicyCausalTransformerConfig,
        image_tokenizer: ImageBaseTokenizer,
        inference_mode: bool = False,
        mask_fn: Optional[Callable] = None,
    ):
        super().__init__()

        self.config = config

        self._construct_transformer()

        self.image_tokenizer = image_tokenizer
        self.text_token_size = self.config.text_token_size
        self.text_tokenizer_embed_dim = self.config.text_tokenizer_embed_dim

        # 1 is the action_out token, the rest are real action tokens
        self.max_seq_len = (
            self.image_tokenizer.get_n_img_tokens()
            + self.text_token_size
            + self.config.n_thinking_tokens
            + 1  # this is the action_out token
            + self.config.n_action_tokens
        ) * config.n_steps
        self._transformer.max_seq_len = self.max_seq_len

        ## TODO: need to double check, this might be wrong even though passed test.
        self.n_tokens_to_first_action = (
            self.image_tokenizer.get_n_img_tokens()
            + self.text_token_size
            + self.config.n_thinking_tokens
            + 1
        )

        if mask_fn is None:
            mask_fn = _img_policy_causal_mask
        self._mask_fn = mask_fn

        self.embedding_std = 0.05

        self.text_embed_mlp = nn.Linear(
            self.text_tokenizer_embed_dim, config.embed_dim, bias=False
        )

        # Img token position makers.
        self.img_pos_tokens = nn.Parameter(
            torch.empty(
                1,
                self.image_tokenizer.get_n_img_tokens(),
                config.embed_dim,
                dtype=torch.bfloat16,
            )
        )
        torch.nn.init.normal_(self.img_pos_tokens, mean=0.0, std=self.embedding_std)

        self.action_pos_tokens = nn.Parameter(
            torch.empty(
                1,
                self.config.n_action_tokens,
                config.action_decoder.embed_dim,
                dtype=torch.bfloat16,
            )
        )

        torch.nn.init.normal_(self.action_pos_tokens, mean=0.0, std=self.embedding_std)

        self.text_pos_tokens = nn.Parameter(
            torch.empty(
                1,
                self.text_token_size,
                config.embed_dim,
                dtype=torch.bfloat16,
            )
        )
        self.text_embedding_for_no_text_input = nn.Parameter(
            torch.empty(
                1,
                self.text_token_size,
                config.embed_dim,
                dtype=torch.bfloat16,
            )
        )
        torch.nn.init.normal_(
            self.text_embedding_for_no_text_input, mean=0.0, std=self.embedding_std
        )
        torch.nn.init.normal_(self.text_pos_tokens, mean=0.0, std=self.embedding_std)

        # Thinking token position makers.
        self.thinking_pos_tokens = nn.Parameter(
            torch.empty(
                1, self.config.n_thinking_tokens, config.embed_dim, dtype=torch.bfloat16
            )
        )
        torch.nn.init.normal_(
            self.thinking_pos_tokens, mean=0.0, std=self.embedding_std
        )

        self.action_out_token = nn.Parameter(
            torch.empty(1, 1, config.embed_dim, dtype=torch.bfloat16)
        )
        torch.nn.init.normal_(self.action_out_token, mean=0.0, std=self.embedding_std)

        self.action_decoder = ActionDecoder(
            cfg=config.action_decoder,
        )

        # We pre-compute the indices of the transformer output that corresponds to each action token.
        output_action_token_idx = []
        for i in range(self.config.n_steps):
            step_start_idx = i * (
                self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
                + 1
                + self.config.n_action_tokens
            )
            action_idx = (
                step_start_idx
                + self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
            )
            output_action_token_idx.append(action_idx)
        self.register_buffer(
            "output_action_token_idx",
            torch.tensor(output_action_token_idx, dtype=torch.long),
            persistent=False,
        )

        self._transformer.construct_transformer_layers()

        # After constructing layers, assign appropriate masks to each layer
        self._assign_layer_masks()

    def _construct_transformer(self):
        self._transformer = Transformer(config=self.config)

    def block_mask_to_device(self, device):
        for layer in self._transformer.transformer_layers:
            if layer.self_attention.block_mask is not None:
                layer.self_attention.block_mask = layer.self_attention.block_mask.to(
                    device
                )

    def rebuild_rope_cache(self, inference_seq_len: int):
        self._transformer.rebuild_rope_cache(inference_seq_len)

    def init_kv_cache_state(self):
        return self._transformer.init_kv_cache_state()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def kv_cache(self):
        return self._transformer.kv_cache

    def _assign_layer_masks(self):
        """
        Assign layer-specific masks based on the per-layer allowed frame counts specified in
        self.config.attention_history_len
        """
        assert len(self.config.attention_history_len) == len(
            self._transformer.transformer_layers
        ), "Number of transformer layers and list of attention n frames should match"
        assert max(self.config.attention_history_len) == self.config.n_steps, (
            "Max attention history len should be equal to the n_steps"
        )
        self._transformer._layer_mask_fns = []

        for i, layer in enumerate(self._transformer.transformer_layers):
            if not isinstance(layer, TransformerSelfAttentionLayer):
                self._layer_mask_fns.append(None)
                continue
            # get allowed frames for this layer.
            allowed_frames = self.config.attention_history_len[i]
            layer_mask_fn = self._mask_fn(
                layer_idx=i,
                n_img_tokens=self.image_tokenizer.get_n_img_tokens(),
                n_thinking_tokens=self.config.n_thinking_tokens,
                n_action_tokens=1
                + self.config.n_action_tokens,  # 1 is the action_out token, the rest are real action tokens
                history_len=allowed_frames,
                n_kv_sink_tokens=self.config.n_kv_sink_tokens,
                n_text_tokens=self.text_token_size,
            )
            self._transformer._layer_mask_fns.append(layer_mask_fn)

            layer.self_attention.block_mask = fa.create_block_mask(
                layer_mask_fn,
                B=None,
                H=None,
                Q_LEN=self.max_seq_len,
                KV_LEN=self.max_seq_len + self.config.n_kv_sink_tokens,
                BLOCK_SIZE=self.config.mask_block_size,
                device=self.device,
            )

    @property
    def step_size(self):
        return self.max_seq_len // self.config.n_steps

    def setup_kv_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.bfloat16
    ):
        # For now at least we assume the length and head is the same at every layer so we only need on kv cache
        # (but separate state for each layer).
        self._transformer.kv_cache = RollingStepKVCache(
            batch_size=batch_size,
            step_size=self.step_size,
            max_T=self.config.n_steps,
            num_kv_heads=self.config.n_kv_head,
            embed_size_per_head=self.config.embed_dim // self.config.n_kv_head,
            device=device,
            dtype=dtype,
        )
        # Tell all the self attention layers about the kv cache object
        # Note this object does not contain kv state, just config for manipulating the cache.
        for i, layer in enumerate(self._transformer.transformer_layers):
            if not isinstance(layer, TransformerSelfAttentionLayer):
                continue

            sa = layer.self_attention
            sa.kv_cache = self._transformer.kv_cache
            sa._mask_fn = self._transformer._layer_mask_fns[i]
            # Precompute decode masks only if we have a block mask and a concrete mask fn
            if sa.block_mask is not None and sa._mask_fn is not None:
                sa.precompute_decode_masks(
                    q_seq_length=self.n_tokens_to_first_action
                    + self.config.n_action_tokens
                )

    def _impute_no_text_embedding(
        self, text_tokens_embed: torch.Tensor
    ) -> torch.Tensor:
        BxT, action_tokens, dim = text_tokens_embed.shape
        text_tokens_embed_reshaped = text_tokens_embed.reshape(BxT, -1)
        no_text_input_mask = ~torch.any(text_tokens_embed_reshaped, dim=1)
        broadcast_no_text_input_mask = no_text_input_mask.reshape(BxT, 1, 1)
        imputed_text_tokens_embed = torch.where(
            broadcast_no_text_input_mask,
            self.text_embedding_for_no_text_input,
            text_tokens_embed,
        )
        return imputed_text_tokens_embed.reshape(BxT, action_tokens, dim)

    def online_forward(
        self,
        img: torch.Tensor,
        text_tokens_embed: torch.Tensor,
        idx: torch.Tensor,
        kv_cache_state: Optional[List[KVCacheState]] = None,
        should_grow_cache: bool = None,
        action_sampler: Callable = None,
        empty_sampled_action_fn: Callable = None,
        reshape_structured_action_fn: Callable = None,
        action_in_to_tokens_fn: Callable = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This is forward pass with kv cache enabled, it has quite bit of extra logic we keep out of the training forward pass.

        This passes in an image and samples a0. Then call rolling_inference_next_action to sample the rest of the action.
        """
        B, T, C, H, W = img.shape
        # Should be feeding in only 1 frame at a time.
        eager_assert(T, 1)
        eager_assert(B, 1)

        img_tokens = self.image_tokenizer(img)
        eager_assert(
            img_tokens.shape,
            (
                B,
                T,
                self.image_tokenizer.get_n_img_tokens(),
                self.config.embed_dim,
            ),
        )
        eager_assert(
            text_tokens_embed.shape,
            (
                self.text_token_size,
                self.text_tokenizer_embed_dim,
            ),
        )
        img_tokens = img_tokens.view(
            B, self.image_tokenizer.get_n_img_tokens(), self.config.embed_dim
        )
        img_tokens_with_pos = img_tokens + self.img_pos_tokens
        text_tokens_embed = text_tokens_embed.view(
            B, self.text_token_size, self.text_tokenizer_embed_dim
        ).to(self.device)
        text_tokens_embed = self.text_embed_mlp(text_tokens_embed)
        text_tokens_embed = self._impute_no_text_embedding(text_tokens_embed)
        text_tokens_embed_with_pos = text_tokens_embed + self.text_pos_tokens

        # We sample the first action token.
        input_pos = (
            torch.arange(
                +self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
                + 1
                + self.config.n_action_tokens,
                device=idx.device,
                dtype=idx.dtype,
            )
            + idx
        )

        batch_thinking_pos_tokens = self.thinking_pos_tokens.repeat(B, 1, 1)
        batch_action_out_token = self.action_out_token.repeat(B, 1, 1)
        dummy_action_embeddings_in = torch.zeros(
            B,
            self.config.n_action_tokens,
            self.config.embed_dim,
            device=batch_action_out_token.device,
        )

        x = torch.cat(
            [
                text_tokens_embed_with_pos,
                img_tokens_with_pos,
                batch_thinking_pos_tokens,
                batch_action_out_token,
                dummy_action_embeddings_in,
            ],
            dim=1,
        )
        eager_assert(
            x.shape,
            (
                B,
                self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
                + 1
                + self.config.n_action_tokens,
                self.config.embed_dim,
            ),
        )

        y, *_ = self._transformer.forward(
            x,
            input_pos=input_pos,
            kv_cache_state=kv_cache_state,
            should_grow_cache=should_grow_cache,
            use_decode_mask=True,
        )
        eager_assert(
            y.shape,
            (
                B,
                self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
                + 1
                + self.config.n_action_tokens,
                self.config.embed_dim,
            ),
        )

        # The final output token is the a0.
        action_out_position = (
            self.image_tokenizer.get_n_img_tokens()
            + self.text_token_size
            + self.config.n_thinking_tokens
        )
        action_token_out = y[:, action_out_position : action_out_position + 1]
        eager_assert(
            action_token_out.shape,
            (B, 1, self.config.embed_dim),
        )

        # Now we need to decode this action token into an auto-regressive set of actions.
        sampled_action = self.action_decoder.autogressive_sample(
            action_token_out,
            action_sampler,
            empty_sampled_action_fn,
            reshape_structured_action_fn,
            inference_mode=True,
        )
        sampled_action_reshaped = pytree.tree_map(
            lambda x: x.unsqueeze(0), sampled_action
        )
        # update kv cache state with new sampled actions
        action_embeddings_in = action_in_to_tokens_fn(sampled_action_reshaped)
        action_embeddings_in = action_embeddings_in.view(
            B, self.config.n_action_tokens, self.config.embed_dim
        )
        action_embeddings_in_with_pos = action_embeddings_in + self.action_pos_tokens
        x = torch.cat(
            [
                text_tokens_embed_with_pos,
                img_tokens_with_pos,
                batch_thinking_pos_tokens,
                batch_action_out_token,
                action_embeddings_in_with_pos,
            ],
            dim=1,
        )
        eager_assert(
            x.shape,
            (
                B,
                self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
                + 1
                + self.config.n_action_tokens,
                self.config.embed_dim,
            ),
        )
        y_new, kv_cache_state, *_ = self._transformer.forward(
            x,
            input_pos=input_pos,
            kv_cache_state=kv_cache_state,
            should_grow_cache=should_grow_cache,
            use_decode_mask=True,
        )
        eager_assert(
            y_new.shape,
            (
                B,
                self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
                + 1
                + self.config.n_action_tokens,
                self.config.embed_dim,
            ),
        )
        next_idx = input_pos[-1] + 1

        return sampled_action, next_idx, kv_cache_state

    def forward(
        self,
        img: torch.Tensor,
        action_embeddings_in: torch.Tensor,
        text_tokens_embed: torch.Tensor,
        should_grow_cache: bool = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        """
        output:
        - action_out: (B, self.config.n_steps, self.config.embed_dim): the output actions embeddings
        - action_out_tokens: (B, self.config.n_steps, self.config.embed_dim): the output of the starting action token embedding, this is only used for IDM model inference for now.
        """
        B, T, *_ = img.shape
        img_tokens = self.image_tokenizer(img)
        eager_assert(
            text_tokens_embed.shape,
            (
                B,
                T,
                self.text_token_size,
                self.text_tokenizer_embed_dim,
            ),
        )
        eager_assert(
            img_tokens.shape,
            (
                B,
                T,
                self.image_tokenizer.get_n_img_tokens(),
                self.config.embed_dim,
            ),
        )
        eager_assert(
            action_embeddings_in.shape,
            (
                B,
                T,
                self.config.n_action_tokens,
                self.config.embed_dim,
            ),
        )
        img_tokens_with_pos = img_tokens + self.img_pos_tokens.unsqueeze(0)

        action_tokens_with_pos = (
            action_embeddings_in + self.action_pos_tokens.unsqueeze(0)
        )
        text_tokens_embed = self.text_embed_mlp(text_tokens_embed)
        text_tokens_embed = text_tokens_embed.reshape(
            B * T, self.text_token_size, self.config.embed_dim
        )
        text_tokens_embed = self._impute_no_text_embedding(text_tokens_embed)
        text_tokens_embed = text_tokens_embed.reshape(
            B, T, self.text_token_size, self.config.embed_dim
        )
        text_embeddings_with_pos = text_tokens_embed + self.text_pos_tokens

        # Ok, now we are ready to concat the input together of img, thinking, action
        x = []
        # Need to repeat the thinking token along the batch dimension.
        batch_thinking_pos_tokens = self.thinking_pos_tokens.repeat(B, 1, 1)
        eager_assert(
            batch_thinking_pos_tokens.shape,
            (
                B,
                self.config.n_thinking_tokens,
                self.config.embed_dim,
            ),
        )
        batch_action_out_pos_token = self.action_out_token.repeat(B, 1, 1)
        eager_assert(
            batch_action_out_pos_token.shape,
            (B, 1, self.config.embed_dim),
        )

        for i in range(self.config.n_steps):
            this_img = img_tokens_with_pos[:, i]
            this_actions = action_tokens_with_pos[:, i]
            this_text = text_embeddings_with_pos[:, i]
            eager_assert(
                this_img.shape,
                (
                    B,
                    self.image_tokenizer.get_n_img_tokens(),
                    self.config.embed_dim,
                ),
            )
            eager_assert(
                this_actions.shape,
                (
                    B,
                    self.config.n_action_tokens,
                    self.config.embed_dim,
                ),
            )
            eager_assert(
                this_text.shape,
                (B, self.text_token_size, self.config.embed_dim),
            )

            this_step_in = torch.cat(
                [
                    this_text,
                    this_img,
                    batch_thinking_pos_tokens,
                    batch_action_out_pos_token,
                    this_actions,
                ],
                dim=1,
            )
            eager_assert(
                this_step_in.shape,
                (
                    B,
                    self.image_tokenizer.get_n_img_tokens()
                    + self.text_token_size
                    + self.config.n_thinking_tokens
                    + 1
                    + self.config.n_action_tokens,
                    self.config.embed_dim,
                ),
            )

            x.append(this_step_in)

        x = torch.cat(x, dim=1)

        eager_assert(
            x.shape,
            (B, self.max_seq_len, self.config.embed_dim),
        )
        y, _, auxiliary_losses, auxiliary_outputs = self._transformer.forward(
            x, input_pos=input_pos, should_grow_cache=should_grow_cache
        )
        eager_assert(
            y.shape,
            (B, self.max_seq_len, self.config.embed_dim),
        )

        # Now select the output action tokens.
        action_out_tokens = torch.index_select(
            y, dim=1, index=self.output_action_token_idx
        )
        eager_assert(
            action_out_tokens.shape,
            (
                B,
                self.config.n_steps,
                self.config.embed_dim,
            ),
        )

        action_out = self.action_decoder(
            action_embeddings_in=action_embeddings_in,
            input_action_token=action_out_tokens,
        )
        return action_out, action_out_tokens, auxiliary_losses, auxiliary_outputs


class MoEPolicyCausalTransformer(PolicyCausalTransformer):
    def _construct_transformer(self):
        self._transformer = MoETransformer(config=self.config)
