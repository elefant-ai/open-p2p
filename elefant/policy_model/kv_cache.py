import torch
from typing import Optional, NamedTuple


class KVCacheState(NamedTuple):
    k_cache: torch.Tensor
    v_cache: torch.Tensor


class BaseKVCache(torch.nn.Module):
    def __init__(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        num_kv_heads: int = 1,
        embed_size_per_head: int = 128,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.num_kv_heads = num_kv_heads
        self.embed_size_per_head = embed_size_per_head
        self.cache_shape = (batch_size, num_kv_heads, 0, embed_size_per_head)

    def init_state(self, batch_size: Optional[int] = None) -> KVCacheState:
        raise NotImplementedError("Subclasses must implement init_state")

    def update(
        self, state: KVCacheState, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> KVCacheState:
        raise NotImplementedError("Subclasses must implement update")


class RollingStepKVCache(BaseKVCache):
    """Module for caching key-value pairs for the policy transformer model.

    The kv-cache rolls off the oldest step when the cache is full.
    """

    def __init__(
        self,
        batch_size: int,
        step_size: int,
        max_T: int,
        num_kv_heads: int,
        embed_size_per_head: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            num_kv_heads=num_kv_heads,
            embed_size_per_head=embed_size_per_head,
        )

        self.step_size = step_size
        self.max_T = max_T

        self.max_seq_len = step_size * max_T

    def init_state(self, batch_size: Optional[int] = None) -> KVCacheState:
        if batch_size is not None:
            effective_shape = (batch_size, *self.cache_shape[1:])
            self.cache_shape = effective_shape
        k_cache = torch.zeros(self.cache_shape, dtype=self.dtype, device=self.device)
        v_cache = torch.zeros(self.cache_shape, dtype=self.dtype, device=self.device)
        torch._dynamo.mark_dynamic(k_cache, 2)
        torch._dynamo.mark_dynamic(v_cache, 2)

        return KVCacheState(k_cache=k_cache, v_cache=v_cache)

    def update(
        self,
        state: KVCacheState,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        should_grow_cache: Optional[bool] = None,
    ) -> KVCacheState:
        """Update the KV cache with new key-value pairs.

        Args:
            state (KVCacheState): The current state of the KV cache.
            k_val (torch.Tensor): The new key values to add to the cache.
            v_val (torch.Tensor): The new value values to add to the cache.
            should_grow_cache (bool, optional): Whether to grow the cache. If None, it will
                grow if the cache is full. Defaults to None. This flag is set in the online kv
                predict function in order to simplify exporting the model with TensorRT.
        Returns:
            KVCacheState: The updated state of the KV cache.
        """
        if should_grow_cache is None:
            if state.k_cache.shape[2] >= self.max_seq_len:
                k_cache = state.k_cache[:, :, self.step_size :, :]
                v_cache = state.v_cache[:, :, self.step_size :, :]
            else:
                k_cache = state.k_cache
                v_cache = state.v_cache
        elif should_grow_cache:
            k_cache = state.k_cache
            v_cache = state.v_cache
        else:
            k_cache = state.k_cache[:, :, self.step_size :, :]
            v_cache = state.v_cache[:, :, self.step_size :, :]

        k_cache = torch.cat([k_cache, k_val], dim=2)
        v_cache = torch.cat([v_cache, v_val], dim=2)

        return KVCacheState(k_cache=k_cache, v_cache=v_cache)


class AccumulatingKVCache(BaseKVCache):
    """Module for caching key-value pairs for the policy transformer model.

    The kv-cache accumulates new key-value pairs until reset.
    """

    def __init__(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        num_kv_heads: int = 1,
        embed_size_per_head: int = 128,
    ):
        super().__init__(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            num_kv_heads=num_kv_heads,
            embed_size_per_head=embed_size_per_head,
        )

    def init_state(self, batch_size: Optional[int] = None) -> KVCacheState:
        if batch_size is not None:
            effective_shape = (batch_size, *self.cache_shape[1:])
            self.cache_shape = effective_shape
        k_cache = torch.zeros(self.cache_shape, dtype=self.dtype, device=self.device)
        v_cache = torch.zeros(self.cache_shape, dtype=self.dtype, device=self.device)
        return KVCacheState(k_cache=k_cache, v_cache=v_cache)

    def update(
        self,
        state: KVCacheState,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        should_grow_cache: Optional[bool] = None,
    ) -> KVCacheState:
        assert should_grow_cache is None, (
            "AccumulatingKVCache does not support this being set explicitly."
        )
        k_cache = torch.cat([state.k_cache, k_val], dim=2)
        v_cache = torch.cat([state.v_cache, v_val], dim=2)

        return KVCacheState(k_cache=k_cache, v_cache=v_cache)
