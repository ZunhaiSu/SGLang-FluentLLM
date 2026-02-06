# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()


if _is_cuda:
    try:
        from flashinfer import FusedSetKVBufferArg
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to import FusedSetKVBufferArg from flashinfer: {e}")
        raise


def enable_fused_set_kv_buffer() -> bool:

    if not _is_cuda:
        return False

    kv_cache_dtype_str = global_server_args_dict.get("kv_cache_dtype", "auto")

    # CLI currently only supports: auto / fp8_e5m2 / fp8_e4m3
    # When set to auto, KV cache dtype follows model dtype; during model construction, loader temporarily sets
    # torch default dtype to model dtype, therefore torch.get_default_dtype() in __init__
    # can be considered equivalent to model dtype.
    if kv_cache_dtype_str != "auto":
        return False

    return torch.get_default_dtype() == torch.bfloat16


def create_fused_set_kv_buffer_arg(
    value: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    layer_id = layer.layer_id
    token_to_kv_pool = forward_batch.token_to_kv_pool

    k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
    v_buffer = token_to_kv_pool.get_value_buffer(layer_id)

    is_mla = isinstance(token_to_kv_pool, MLATokenToKVPool)

    if is_mla:
        kv_lora_rank = token_to_kv_pool.kv_lora_rank
        k_buffer = k_buffer[..., kv_lora_rank:].view(k_buffer.shape[0], -1)
        v_buffer = v_buffer[..., :kv_lora_rank].view(v_buffer.shape[0], -1)
    else:
        k_buffer = k_buffer.view(k_buffer.shape[0], -1)
        v_buffer = v_buffer.view(v_buffer.shape[0], -1)

    return FusedSetKVBufferArg(
        value=value,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_scale=layer.k_scale,
        v_scale=layer.v_scale,
        cache_loc=forward_batch.out_cache_loc,
    )