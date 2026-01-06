import torch
import torch.nn as nn
from torch.nn.attention import flex_attention as fa
from torch.nn.attention.flex_attention import BlockMask
from torch.nn import functional as F
from typing import Optional
from elefant.policy_model.pos_embed import RotaryPositionalEmbeddings
import logging
from elefant.config import ConfigBase
from elefant.policy_model.kv_cache import KVCacheState
from typing import Tuple, List
from elefant.torch import eager_assert
from elefant.modules import RMSNormF32
from collections import defaultdict
from typing import Callable, Literal
from elefant.modules.moe import MoE, TokenChoiceTopKRouter, GroupedExperts


def get_mask_mod(mask_mod, offset):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod


def get_decode_mask(
    block_mask: BlockMask,
    offset,
    mask_fn,
    q_seq_length,
    kv_seq_length,
    device,
):
    """Get the mask for decoding at a specific position"""
    # block index for query token to select which block the flex attention should select
    # for query tokens(i.e. select relative query block from block mask)
    block_index = torch.tensor(
        [offset // block_mask.BLOCK_SIZE[0]], dtype=torch.int32, device=device
    )
    mask = block_mask[:, :, block_index]
    # get a mask function for the particular offset during decoding.
    # i.e. originally mask_fn is the original mask_fn we passed to create block mask
    # but since we are decoding and q's length is 3(or 1), torch calls the original mask_fn
    # with the 0, 1, 2 indices, we add an offset for this so flex attention realizes
    # that the indices are not 0, 1, 2 but the starting indices + the offset(absolute position up to kv cache "fullness")
    mask.mask_mod = get_mask_mod(mask_fn, offset)
    # This is done so we get a valid subset of the mask during decoding
    mask.seq_lengths = (q_seq_length, kv_seq_length)
    return mask


class SparseMoEConfig(ConfigBase):
    num_experts: int = 16
    experts_per_token: int = 4
    inference_mode: bool = False


class TransformerConfig(ConfigBase):
    n_transformer_layers: int = 3
    embed_dim: int = 128
    dropout: float = 0.1
    n_q_head: int = 8
    n_kv_head: int = 8
    model_type: Literal["dense", "sparse_moe"] = "dense"


class ImageGenerationTransformerConfig(TransformerConfig):
    n_steps: int = 3
    n_thinking_tokens: int = 1
    skip_steps: int = 0
    mask_block_size: int = 128
    n_image_tokens: int = 1
    n_codebook_size: int = 16
    ## this number is different from what we
    ## used in IDM model, IDM model used 1
    ## regardless of the num of discrete latents
    ## to avoid autorergession generation
    n_latent_action_tokens: int = 1


class LatentActionTransformerConfig(ImageGenerationTransformerConfig):
    latent_action_embed_dim: int


class SelfAttention(nn.Module):
    # Useful reference:
    # https://github.com/pytorch/torchtune/blob/main/torchtune/modules/attention.py

    def __init__(
        self,
        input_dim: int,
        embed_size_per_head: int,
        n_q_head: int,
        n_kv_head: int,
        dropout: float = 0.0,
        pos_embedding=None,
        is_causal: bool = False,
        n_kv_sink_tokens: int = 1,
    ):
        """Self-attention layer.

        There are 3 options for masking:
            - A dense mask.
            - A flex attention mask.
            - No mask, pass is_causal=True.
        """
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.input_dim = input_dim
        self.is_causal = is_causal
        self.embed_size_per_head = embed_size_per_head
        self.head_embed_size = embed_size_per_head * n_q_head
        self.k_norm = RMSNormF32(dim=self.head_embed_size, eps=1e-5)
        self.q_norm = RMSNormF32(dim=self.head_embed_size, eps=1e-5)
        if self.head_embed_size != self.input_dim:
            logging.warning(
                "Typically self-attention has the same dimensionality on input and output."
                + f"input_dim={input_dim} output_dim=embed_size_per_head * n_q_head = {self.head_embed_size}"
            )
        self.kv_embed_size = embed_size_per_head * n_kv_head
        self.c_attn = nn.Linear(
            input_dim, 2 * self.kv_embed_size + self.head_embed_size, bias=False
        )
        # output projection
        self.c_proj = nn.Linear(self.head_embed_size, input_dim, bias=False)

        # regularization
        self.residual_dropout = nn.Dropout(dropout)
        self.n_q_head = n_q_head
        self.n_kv_head = n_kv_head
        assert n_kv_head <= n_q_head
        self.grouped_qa = n_q_head != n_kv_head
        self.dropout = dropout

        self.pos_embedding = pos_embedding
        self.kv_cache = None
        self.block_mask = None
        self._mask_fn = None
        self._cached_decode_mask = None

        # trainable KV sink tokens
        self.n_kv_sink_tokens = n_kv_sink_tokens
        assert self.n_kv_sink_tokens >= 0
        if self.n_kv_sink_tokens > 0:
            self.k_sinks = nn.Parameter(
                torch.randn(
                    1,
                    self.n_kv_head,
                    self.n_kv_sink_tokens,
                    self.embed_size_per_head,
                    dtype=torch.bfloat16,
                )
            )
            self.v_sinks = nn.Parameter(
                torch.randn(
                    1,
                    self.n_kv_head,
                    self.n_kv_sink_tokens,
                    self.embed_size_per_head,
                    dtype=torch.bfloat16,
                )
            )
        else:
            self.k_sinks = None
            self.v_sinks = None

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # need to do like this, otherwise to doesn't move block mask for some reason
        device = next(self.parameters()).device
        if self.block_mask is not None:
            self.block_mask = self.block_mask.to(device)
        if self._cached_decode_mask is not None:
            self._cached_decode_mask = self._cached_decode_mask.to(device)
        return self

    def precompute_decode_masks(self, q_seq_length: int = 3):
        self._cached_decode_mask = None
        self.stable_start = None

        if self.block_mask is None or self._mask_fn is None or self.kv_cache is None:
            return
        self.stable_start = self.kv_cache.max_seq_len - self.kv_cache.step_size
        kv_seq_length = self.kv_cache.max_seq_len + self.n_kv_sink_tokens

        self._cached_decode_mask = get_decode_mask(
            block_mask=self.block_mask,
            offset=self.stable_start,
            mask_fn=self._mask_fn,
            q_seq_length=q_seq_length,
            kv_seq_length=kv_seq_length,
            device=self.device,
        )

    def forward(
        self,
        x: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        flex_attention_mask: Optional[torch.Tensor] = None,
        kv_cache_state: Optional[KVCacheState] = None,
        should_grow_cache: bool = None,
        use_decode_mask: bool = False,
    ):
        B, T, D = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        eager_assert(D, self.input_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(
            [self.head_embed_size, self.kv_embed_size, self.kv_embed_size], dim=2
        )
        if self.q_norm is not None and self.k_norm is not None:
            _dtype = k.dtype
            q = self.q_norm(q).to(_dtype)
            k = self.k_norm(k).to(_dtype)

        q = q.view(B, T, self.n_q_head, self.embed_size_per_head)
        if self.pos_embedding is not None:
            # TODO: we could use the step no as the input embedding.
            q = self.pos_embedding(q, input_pos=input_pos)
        q = q.transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.embed_size_per_head)
        if self.pos_embedding is not None:
            k = self.pos_embedding(k, input_pos=input_pos)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.embed_size_per_head)
        v = v.transpose(1, 2)

        if kv_cache_state is not None:
            kv_cache_state = self.kv_cache.update(
                kv_cache_state, k, v, should_grow_cache
            )
            k = kv_cache_state.k_cache
            v = kv_cache_state.v_cache

        # Prepend KV sinks if enabled
        if self.n_kv_sink_tokens != 0 and k.shape[-2] != 0:
            sink_k = self.k_sinks
            sink_v = self.v_sinks

            k = torch.cat([sink_k.expand(B, -1, -1, -1), k], dim=2)
            v = torch.cat([sink_v.expand(B, -1, -1, -1), v], dim=2)

        if flex_attention_mask is not None:
            # Assert that we're not using torch.compile with CPU when using flex_attention
            if q.device.type == "cpu" and torch.compiler.is_compiling():
                raise AssertionError(
                    "flex_attention with torch.compile is buggy on CPU."
                )

            eager_assert(self.is_causal, False)

            q_len = q.size(2)
            kv_len = k.size(2)

            if use_decode_mask and self._mask_fn is not None:
                # Use the precomputed steady-state mask when the cache is full
                if (
                    self.kv_cache is not None
                    and kv_len == (self.kv_cache.max_seq_len + self.n_kv_sink_tokens)
                    and self._cached_decode_mask is not None
                ):
                    flex_attention_mask = self._cached_decode_mask
                else:
                    # q_start_pos is in the KV+S space; convert to Q (no-sinks) by subtracting S
                    q_start_pos = kv_len - q_len
                    offset = q_start_pos - self.n_kv_sink_tokens
                    flex_attention_mask = get_decode_mask(
                        block_mask=flex_attention_mask,
                        offset=offset,
                        mask_fn=self._mask_fn,
                        q_seq_length=q_len,
                        kv_seq_length=kv_len,
                        device=q.device,
                    )

            y = fa.flex_attention(
                q,
                k,
                v,
                block_mask=flex_attention_mask,
                enable_gqa=self.grouped_qa,
            )
        else:
            eager_assert(self.is_causal, True)

            # Construct a standard causal mask, trimming as needed when using a kv cache.
            q_len = q.size(2)
            kv_len = k.size(2)
            # causal on real tokens, allow attending to sinks
            real_kv_len = kv_len - self.n_kv_sink_tokens
            causal = torch.tril(
                torch.ones(real_kv_len, real_kv_len, dtype=torch.bool, device=q.device)
            )[-q_len:, :]
            if self.n_kv_sink_tokens != 0:
                sinks_allow = torch.ones(
                    q_len, self.n_kv_sink_tokens, dtype=torch.bool, device=q.device
                )
                attn_mask = torch.cat([sinks_allow, causal], dim=1)
            else:
                attn_mask = causal

            y = F.scaled_dot_product_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                enable_gqa=self.grouped_qa,
                attn_mask=attn_mask,
            )

        eager_assert(y.shape, (B, self.n_q_head, T, self.embed_size_per_head))

        # This has to be contiguous otherwise we get an error when using torch.compile.
        # -1 dimension is T, but trying to make torch.compile happy.
        # This should be able to be a view (and reshape should only make a view normally)
        # but for tensorrt it complains so made it a reshape for now.
        y = y.transpose(1, 2).contiguous().reshape(B, T, self.head_embed_size)

        # output projection
        y = self.c_proj(y)
        if self.dropout > 0.0:
            y = self.residual_dropout(y)

        return y, kv_cache_state


class PackedSwiGLUFFN(nn.Module):
    # Adapted from https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    def __init__(
        self,
        dim: int,
        hidden_dim: int,  # Traditional 4*dim
        multiple_of: int,
    ):
        """
        Efficient SwiGLU implementation. The hidden_dim is adjusted to be a multiple of multiple_of.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim

        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        return (
            self.w2(F.silu(x1) * x3),
            {},
            {"num_tokens_per_expert": torch.tensor([1.0])},
        )


class TransformerSelfAttentionLayer(nn.Module):
    def __init__(self, dim: int, self_attention: SelfAttention, ffn: PackedSwiGLUFFN):
        super().__init__()
        self.self_attention = self_attention
        self.ffn = ffn
        self.norm_eps = 1e-5

        self.dim = dim
        self.self_attention_norm = RMSNormF32(dim=dim, eps=self.norm_eps)
        self.ffn_norm = RMSNormF32(dim=dim, eps=self.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        flex_attention_mask: Optional[torch.Tensor] = None,
        kv_cache_state: Optional[KVCacheState] = None,
        should_grow_cache: bool = None,
        use_decode_mask: bool = False,
    ):
        h = self.self_attention_norm(x)
        attn_out, kv_cache_state = self.self_attention(
            h,
            input_pos=input_pos,
            flex_attention_mask=flex_attention_mask,
            kv_cache_state=kv_cache_state,
            should_grow_cache=should_grow_cache,
            use_decode_mask=use_decode_mask,
        )
        eager_assert(attn_out.shape, x.shape)

        # Residual connection.
        h2 = attn_out + x

        mlp_inp = self.ffn_norm(h2)
        ffn_out, auxiliary_losses, auxiliary_outputs = self.ffn(mlp_inp)
        eager_assert(ffn_out.shape, x.shape)

        # Residual connection.
        out = h2 + ffn_out

        return out, kv_cache_state, auxiliary_losses, auxiliary_outputs


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    def rebuild_rope_cache(self, inference_seq_len: int):
        """For inference we can handle longer sequences than we trained on.
        This method rebuilds the rope cache for the new sequence length.
        """
        self.pos_embeddings.build_rope_cache(inference_seq_len)

    def construct_transformer_layers(self):
        """Override this to modify the transformer layers.

        Call this after the init.
        """
        embed_size_per_head = self.config.embed_dim // self.config.n_q_head
        assert embed_size_per_head % 2 == 0, "rotary embedding dim must be even"
        assert self.config.embed_dim % self.config.n_q_head == 0
        self.pos_embeddings = RotaryPositionalEmbeddings(
            dim=embed_size_per_head, max_seq_len=self.max_seq_len, base=10_000
        )

        transformer_layers = []
        for i in range(self.config.n_transformer_layers):
            sa = self.construct_self_attention()
            ffn = PackedSwiGLUFFN(
                dim=self.config.embed_dim,
                hidden_dim=self.config.embed_dim * 4,
                multiple_of=8,
            )
            layer = TransformerSelfAttentionLayer(
                dim=self.config.embed_dim,
                self_attention=sa,
                ffn=ffn,
            )
            transformer_layers.append(layer)

        self.transformer_layers = nn.ModuleList(transformer_layers)

    def construct_self_attention(self):
        """Override this to modify the self attention block."""
        attention_block = SelfAttention(
            input_dim=self.config.embed_dim,
            embed_size_per_head=self.config.embed_dim // self.config.n_q_head,
            n_q_head=self.config.n_q_head,
            n_kv_head=self.config.n_kv_head,
            dropout=self.config.dropout,
            pos_embedding=self.pos_embeddings,
            n_kv_sink_tokens=self.config.n_kv_sink_tokens,
        )
        return attention_block

    def init_kv_cache_state(self, batch_size: Optional[int] = None):
        cache_state = []
        for layer in self.transformer_layers:
            if isinstance(layer, TransformerSelfAttentionLayer):
                cache_state.append(self.kv_cache.init_state(batch_size))
        return cache_state

    def _process_auxiliary_losses(self, auxiliary_losses: defaultdict) -> dict:
        processed_losses = {}
        for k, v in auxiliary_losses.items():
            if v:
                processed_losses[k] = torch.stack(v).mean()
        return processed_losses

    def _process_auxiliary_outputs(
        self, num_tokens_per_expert_list: list, x_device: torch.device
    ) -> dict:
        auxiliary_outputs = {}
        if not num_tokens_per_expert_list:
            num_tokens_per_expert = torch.tensor([1.0], device=x_device)
        else:
            num_tokens_per_expert = (
                torch.stack(num_tokens_per_expert_list).sum(dim=0).float()
            )
            num_tokens_per_expert /= num_tokens_per_expert.sum()

        auxiliary_outputs["num_tokens_per_expert"] = num_tokens_per_expert
        return auxiliary_outputs

    def forward(
        self,
        x: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache_state: Optional[List[KVCacheState]] = None,
        should_grow_cache: bool = None,
        use_decode_mask: bool = False,
    ):
        if kv_cache_state is not None:
            assert len(kv_cache_state) == len(self.transformer_layers)
            kv_cache_out = []
        else:
            kv_cache_out = None

        auxiliary_losses = defaultdict(list)
        auxiliary_outputs = defaultdict(list)
        num_tokens_per_expert_list = []
        for i, layer in enumerate(self.transformer_layers):
            if isinstance(layer, TransformerSelfAttentionLayer):
                layer_block_mask = layer.self_attention.block_mask
                if layer_block_mask is not None and use_decode_mask:
                    if hasattr(self, "_layer_mask_fns"):
                        layer.self_attention._mask_fn = self._layer_mask_fns[i]
                    else:
                        layer.self_attention._mask_fn = getattr(
                            self, "_base_mask_fn", None
                        )

                if kv_cache_state is not None:
                    this_kv_cache_state = kv_cache_state[i]
                else:
                    this_kv_cache_state = None
                x, this_kv_cache_state, _auxiliary_losses, _auxiliary_outputs = layer(
                    x,
                    input_pos=input_pos,
                    flex_attention_mask=layer_block_mask,
                    kv_cache_state=this_kv_cache_state,
                    should_grow_cache=should_grow_cache,
                    use_decode_mask=use_decode_mask,
                )
                for k, v in _auxiliary_losses.items():
                    auxiliary_losses[k].append(v)
                num_tokens_per_expert_list.append(
                    _auxiliary_outputs["num_tokens_per_expert"].to(x.device)
                )

                if this_kv_cache_state is not None:
                    kv_cache_out.append(this_kv_cache_state)
            else:
                # TODO: not sure if this still works
                x = layer(x, input_pos=input_pos)

        auxiliary_losses = self._process_auxiliary_losses(auxiliary_losses)
        auxiliary_outputs = self._process_auxiliary_outputs(
            num_tokens_per_expert_list, x.device
        )
        return x, kv_cache_out, auxiliary_losses, auxiliary_outputs


class MoETransformer(Transformer):
    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)

    def construct_transformer_layers(self):
        embed_size_per_head = self.config.embed_dim // self.config.n_q_head
        assert embed_size_per_head % 2 == 0, "rotary embedding dim must be even"
        assert self.config.embed_dim % self.config.n_q_head == 0
        self.pos_embeddings = RotaryPositionalEmbeddings(
            dim=embed_size_per_head, max_seq_len=self.max_seq_len, base=10_000
        )

        transformer_layers = []
        for i in range(self.config.n_transformer_layers):
            sa = self.construct_self_attention()
            router = TokenChoiceTopKRouter(
                gate=nn.Linear(
                    self.config.embed_dim,
                    self.config.sparse_moe.num_experts,
                    bias=False,
                ),
                dim=self.config.embed_dim,
                num_experts=self.config.sparse_moe.num_experts,
                experts_per_token=self.config.sparse_moe.experts_per_token,
            )
            experts = GroupedExperts(
                dim=self.config.embed_dim,
                hidden_dim=self.config.embed_dim * 4,
                num_experts=self.config.sparse_moe.num_experts,
                multiple_of=8,
                inference_mode=self.config.sparse_moe.inference_mode,
            )
            shared_expert = None
            ffn = MoE(
                experts=experts,
                router=router,
                shared_expert=shared_expert,
            )
            layer = TransformerSelfAttentionLayer(
                dim=self.config.embed_dim,
                self_attention=sa,
                ffn=ffn,
            )
            transformer_layers.append(layer)

        self.transformer_layers = nn.ModuleList(transformer_layers)
