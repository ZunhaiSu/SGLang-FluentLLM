from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
import torch

from typing import (
    Any,
    NamedTuple,
    Optional,
    Tuple,
)
class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Page indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        device_prefix_length:   Length of the common prefix in tokens (not pages).
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
        host_hit_length :   Number of tokens hit on the host, if applicable.
                            0 if HiCache is not enabled.
                            Note: node.host_value stores token indices.
        mamba_branching_seqlen: The mamba radix cache branching point, which is the longest
                                page-aligned position that could've been cache hit if there
                                exists a mamba state.
    """

    device_indices: torch.Tensor = None
    last_device_node: Any = None
    device_prefix_length: int = 0
    last_host_node: Any = None
    host_hit_length: int = 0
    mamba_branching_seqlen: Optional[int] = None


class BasePrefixCache(ABC):
    """Cache can be indexed by either rid or key."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, **kwargs) -> Tuple[List[int], int]:
        pass

    @abstractmethod
    def insert(self, **kwargs):
        pass

    @abstractmethod
    def cache_finished_req(self, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, **kwargs):
        pass

    @abstractmethod
    def evict(self, num_tokens: int, evict_callback: Callable):
        pass

    @abstractmethod
    def inc_lock_ref(self, node):
        pass

    @abstractmethod
    def dec_lock_ref(self, node):
        pass

    @abstractmethod
    def evictable_size(self):
        pass

    @abstractmethod
    def protected_size(self):
        raise NotImplementedError()

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()
