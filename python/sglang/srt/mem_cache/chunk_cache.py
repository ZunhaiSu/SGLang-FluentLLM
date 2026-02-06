from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.allocator import KVAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.req import Req


class ChunkCacheEntry:
    def __init__(self, rid: str, value: torch.Tensor):
        self.rid = rid
        self.value = value


class ChunkCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        kv_allocator: KVAllocator,
    ):
        self.disable = True
        self.req_to_token_pool = req_to_token_pool
        self.kv_allocator = kv_allocator
        self.entries: Dict[str, ChunkCacheEntry] = {}
        self.reset()

    def reset(self):
        self.entries = {}
        self.draft_entries = {}

    def match_prefix(self, rid: str, key: List[int]) -> Tuple[List[int], ChunkCacheEntry]:
        if rid not in self.entries:
            return [], None

        entry = self.entries[rid]
        max_prefix_len = len(key)
        return entry.value[:max_prefix_len], entry

    def match_prefix_draft(self, rid: str, key: List[int]) -> Tuple[List[int], ChunkCacheEntry]:
        if rid not in self.entries:
            return [], None

        entry = self.draft_entries[rid]
        max_prefix_len = len(key)
        return entry.value[:max_prefix_len], entry

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        if token_ids is None:
            # For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids
            token_id_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        else:
            token_id_len = len(token_ids)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :token_id_len
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.kv_allocator.free(req.req_pool_idx, kv_indices)

        if req.rid in self.entries:
            del self.entries[req.rid]

    def cache_unfinished_req(self, req: Req):
        token_id_len = len(req.fill_ids)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :token_id_len
        ]

        if req.rid not in self.entries:
            self.entries[req.rid] = ChunkCacheEntry(req.rid, kv_indices)

        entry = self.entries[req.rid]
        entry.value = kv_indices
        req.prefix_indices = kv_indices
        req.last_node = entry
        req.pages_info = self.kv_allocator.get_pages_info(req.req_pool_idx)
        req.req_to_token_pool_info = self.req_to_token_pool.get_req_pool_info(req.req_pool_idx)

    def insert(self):
        raise NotImplementedError()

    def evict(self, num_tokens: int, evict_callback: Callable):
        pass

    def inc_lock_ref(self, node):
        return 0

    def dec_lock_ref(self, node):
        return 0

    def evictable_size(self):
        return 0

    def pretty_print(self):
        return ""

    def protected_size(self):
        return 0
