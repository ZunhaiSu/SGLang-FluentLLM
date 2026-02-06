from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import add_prefix, align, is_cuda, is_npu

if is_cuda():
    try:
        import deep_gemm_oss
    except ImportError as e:
        deep_gemm_oss = e
from sglang.srt.layers.attention.dsa.utils import NSA_USE_REAL_INDEXER
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group, get_attention_tp_rank, get_attention_tp_size, get_attn_tp_dp_convertor
)
from sglang.srt.layers.utils import (
    CP_METADATA, cp_all_gather_rerange_output, 
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.layers.attention.dsa_backend import DpskSparseAttnBackend

from sglang.srt.utils import get_colorful_logger
logger = get_colorful_logger(__name__)

DUAL_STREAM_TOKEN_THRESHOLD = 1024 if is_cuda() else 0

class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        lengths: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    if x.shape[0] == 0:
        return x
    from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


class V32LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)


class Indexer(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        index_k_norm_type: str,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream

        self.cp_size = get_attention_tp_size()
        self.cp_rank = get_attention_tp_rank()
        self.comm_convertor = get_attn_tp_dp_convertor()
        assert self.comm_convertor is not None

        if is_cuda():
            self.sm_count = deep_gemm_oss.get_num_sms()
            self.half_device_sm_count = align(self.sm_count // 2, 8)

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.k_norm = V32LayerNorm(self.head_dim)
        # NOTE: weight_proj is not quantized
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=True,
            device="cuda" if not is_npu() else "npu",
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5

    def _forward_fake(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ):
        bs = x.shape[0]
        assert self.index_topk == 2048
        ans = torch.arange(0, self.index_topk, dtype=torch.int32, device=x.device)[
            None, ...
        ].repeat(bs, 1)
        if forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.seq_lens_cpu is not None
            )
            which = 0
            for i, (kv_len, qo_len) in enumerate(
                zip(
                    forward_batch.seq_lens_cpu.tolist(),
                    forward_batch.extend_seq_lens_cpu,
                    strict=True,
                )
            ):
                for j in range(kv_len - qo_len, kv_len):
                    ans[which, j + 1 :] = -1
                    which += 1
            assert which == ans.shape[0]
        else:
            assert forward_batch.seq_lens_cpu is not None
            for i, seq_len in enumerate(forward_batch.seq_lens_cpu.tolist()):
                ans[i, seq_len:] = -1

        return ans

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
        weights, _ = self.weights_proj(x.float())
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def get_head_gate(self, x: torch.Tensor, out: torch.Tensor):
        torch.matmul(x, self.weights_proj.weight.T, out=out)

    def get_q_k(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
    ):
        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

        q_rope, _ = torch.split(
            query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        if q_rope.shape[0] > 0:
            q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
            query[..., : self.rope_head_dim] = q_rope
            key[..., : self.rope_head_dim] = k_rope

        if CP_METADATA:
            key = cp_all_gather_rerange_output(
                key.contiguous(),
                CP_METADATA.value,
                self.comm_convertor
            )

        from fast_hadamard_transform import hadamard_transform_fuse_quant

        qd = query.shape[-1]
        kd = key.shape[-1]
        assert qd == kd and (qd & (qd - 1)) == 0, \
            "Dimension must be a power of 2 for Hadamard transform."
        q_fp8, q_scale, k_fp8, k_scale = hadamard_transform_fuse_quant(query, key, qd**-0.5)

        return q_fp8, q_scale, k_fp8, k_scale

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        assert page_size == 64, "only support page size 64"

        # NOTE(dark): this support extend/decode/decode+graph
        block_tables = metadata.get_page_table_64()

        # align to page_size
        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        blocksize = page_size
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            seqlens_32 = metadata.get_seqlens_expanded()
        else:
            seqlens_32 = metadata.get_seqlens_int32()

        schedule_metadata = deep_gemm_oss.get_paged_mqa_logits_metadata(
            seqlens_32, blocksize, self.sm_count
        )

        assert len(q_fp8.shape) == 3
        # the next_n dim is always 1, and block_tables corresponding to it
        q_fp8 = q_fp8.unsqueeze(1)
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        # [b, max_seq_len]
        logits = deep_gemm_oss.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            seqlens_32,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )
        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = metadata.topk_transform(logits, self.index_topk)
        return topk_result

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_list = []
        q_offset = 0
        k_offset = 0

        block_tables = metadata.get_page_table_64()
        seq_lens_expanded = metadata.get_seqlens_expanded()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            assert isinstance(seq_len, int)
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            seq_len = forward_batch.seq_lens_cpu[i]
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            ks = torch.full((extend_seq_len,), k_offset, dtype=torch.int32, device="cuda")
            ke = ks + seq_lens_expanded[q_offset : q_offset + extend_seq_len]
            k_fp8_list.append(k_fp8)
            k_scale_list.append(k_scale)
            ks_list.append(ks)
            ke_list.append(ke)
            q_offset += extend_seq_len
            k_offset += seq_len

        k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)
        logits = deep_gemm_oss.fp8_mqa_logits(
            q_fp8,      # [s_q, nh, hd]
            kv_fp8,     # tuple: ([s_k, hd], [s_k])
            weights,    # [s_q, nh]
            ks,         # cu_seq_len_k_start, [s_q]
            ke,         # cu_seq_len_k_end, [s_q]
            clean_logits=False, # not clean the unfilled logits into -inf
        )

        assert logits.shape[0] == len(seq_lens_expanded)
        topk_result = metadata.topk_transform(logits, self.index_topk, row_starts=ks)

        return topk_result

    def _get_topk_ragged_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        seq_lens_expanded: torch.Tensor,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_list = []
        q_offset = 0
        k_offset = 0

        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        assert forward_batch.batch_size <= 1, f"CP only support bs<=1 for now"

        seq_len = forward_batch.seq_lens_cpu[0].item()
        k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
            layer_id,
            seq_len,
            block_tables[0],
        )
        k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
            layer_id,
            seq_len,
            block_tables[0],
        )
        # extend_seq_len = forward_batch.extend_seq_lens_cpu[0]
        q_seq_len = q_fp8.shape[0]
        ks = torch.full((q_seq_len,), k_offset, dtype=torch.int32, device="cuda")
        ke = ks + seq_lens_expanded[q_offset : q_offset + q_seq_len]
        k_fp8_list.append(k_fp8)
        k_scale_list.append(k_scale)
        ks_list.append(ks)
        ke_list.append(ke)
        # q_offset += extend_seq_len
        # k_offset += seq_len

        k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)
        logits = deep_gemm_oss.fp8_mqa_logits(
            q_fp8,      # [s_q, nh, hd]
            kv_fp8,     # tuple: ([s_k, hd], [s_k])
            weights,    # [s_q, nh]
            ks,         # cu_seq_len_k_start, [s_q]
            ke,         # cu_seq_len_k_end, [s_q]
            clean_logits=False, # not clean the unfilled logits into -inf
        )

        assert logits.shape[0] == len(seq_lens_expanded)
        topk_result = metadata.topk_transform(
            logits, self.index_topk,
            lengths=seq_lens_expanded,
            cu_seqlens_q=seq_lens_expanded.new_tensor([0, q_seq_len])
        )

        return topk_result

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        index_k: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        comm_manager: Optional[DecoderCommMananger] = None
    ) -> Optional[torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )
        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

        q_rope, _ = torch.split(
            query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        key = index_k
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        if q_rope.shape[0] > 0:
            q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
            query[..., : self.rope_head_dim] = q_rope
            key[..., : self.rope_head_dim] = k_rope

        from fast_hadamard_transform import hadamard_transform_fuse_quant

        qd = query.shape[-1]
        kd = key.shape[-1]
        assert qd == kd and (qd & (qd - 1)) == 0, \
            "Dimension must be a power of 2 for Hadamard transform."
        q_fp8, q_scale, k_fp8, k_scale = hadamard_transform_fuse_quant(query, key, qd**-0.5)

        weights = self._get_logits_head_gate(x, q_scale)

        # k_fp8: (seq_len, head_dim) fp8_e4m3fn
        # k_buffer: (num_total_tokens + page_size, head_dim) fp8_e4m3fn
        # k_scale: (seq_len, head_dim // block_size = 1) fp8_e4m3fn
        # k_scale_cache: (num_total_tokens + page_size, head_dim // block_size = 1) fp8_e4m3fn
        forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(
            layer_id=layer_id,
            loc=forward_batch.out_cache_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

        if x.shape[0] == 0:
            return x.new_zeros((0, self.index_topk), dtype=torch.int32)

        if is_cuda():
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
            else:
                assert forward_batch.seq_lens_cpu is not None
                if CP_METADATA:
                    seqlens_expanded = metadata.get_seqlens_expanded()
                    topk_result = self._get_topk_ragged_cp(
                        forward_batch, layer_id, q_fp8, weights, metadata, seqlens_expanded
                    )
                else:
                    topk_result = self._get_topk_ragged(
                        forward_batch, layer_id, q_fp8, weights, metadata
                    )

        return topk_result

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> torch.Tensor:
        import torch_npu

        from sglang.srt.layers.dp_attention import (
            get_attention_tp_rank,
            get_attention_tp_size,
        )
        from sglang.srt.utils import get_bool_env_var

        if forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = forward_batch.attn_backend.forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = (
                forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int
            )
        enable_index_cp = (
            get_bool_env_var("SGLANG_USE_AG_AFTER_QLORA") and layer_id >= 4
        )
        is_prefill = forward_batch.forward_mode.is_extend()

        attention_tp_rank = get_attention_tp_rank()
        attention_tp_size = get_attention_tp_size()

        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        if is_prefill and enable_index_cp:
            slice_length = cos.shape[0] // attention_tp_size
            cos = cos[
                slice_length
                * attention_tp_rank : slice_length
                * (attention_tp_rank + 1)
            ]
            sin = sin[
                slice_length
                * attention_tp_rank : slice_length
                * (attention_tp_rank + 1)
            ]

        slot_mapping = forward_batch.out_cache_loc
        block_table = forward_batch.attn_backend.forward_metadata.block_tables

        bs = x.shape[0]

        q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
        q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
        q_pe, q_nope = torch.split(
            q,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64, 64 + 64]

        q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin).view(
            bs, self.n_heads, self.rope_head_dim
        )  # [bs, n, d]
        q = torch.cat([q_pe, q_nope], dim=-1)

        k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
        k = self.k_norm(k_proj)
        k_pe, k_nope = torch.split(
            k,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64 + 64]

        k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin).view(
            bs, 1, self.rope_head_dim
        )  # [bs, 1, d]
        k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        if is_prefill and enable_index_cp:
            k, local_k = (
                torch.empty(
                    (k.shape[0] * attention_tp_size, k.shape[1], k.shape[2]),
                    dtype=k.dtype,
                    device=k.device,
                ),
                k,
            )
            get_attention_tp_group().all_gather_into_tensor(k, local_k)

        forward_batch.token_to_kv_pool.set_index_k_buffer(layer_id, slot_mapping, k)

        indexer_input = {}
        if is_prefill:
            actual_seq_lengths_kv = forward_batch.seq_lens.to(device=q.device)
            actual_seq_lengths_q = forward_batch.seq_lens.cumsum(dim=0).to(
                device=q.device
            )
            if enable_index_cp:
                actual_seq_lengths_q -= bs * attention_tp_rank
                actual_seq_lengths_q = torch.max(
                    actual_seq_lengths_q,
                    torch.zeros_like(actual_seq_lengths_q).to(
                        device=actual_seq_lengths_q.device
                    ),
                )
                actual_seq_lengths_q = torch.min(
                    actual_seq_lengths_q,
                    torch.full(actual_seq_lengths_q.shape, bs).to(
                        device=actual_seq_lengths_q.device
                    ),
                )

        else:
            if forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q is None:
                actual_seq_lengths_q = torch.tensor(
                    [1 + i * 1 for i in range(bs)], dtype=torch.int32, device=k.device
                )
            else:
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )

        past_key_states = forward_batch.token_to_kv_pool.get_index_k_buffer(layer_id)

        x = x.view(-1, self.hidden_size)
        weights = self.weights_proj(x)[0]
        block_table = (
            block_table[: actual_seq_lengths_q.size()[0]] if is_prefill else block_table
        )

        topk_indices = torch.ops.custom.npu_lightning_indexer(
            query=q.view(-1, self.n_heads, self.head_dim),
            key=past_key_states,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
            actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(torch.int32),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )

        if is_prefill and enable_index_cp:
            topk_indices, local_topk_indices = (
                torch.empty(
                    (
                        topk_indices.shape[0] * attention_tp_size,
                        topk_indices.shape[1],
                        topk_indices.shape[2],
                    ),
                    dtype=topk_indices.dtype,
                    device=topk_indices.device,
                ),
                topk_indices,
            )
            get_attention_tp_group().all_gather_into_tensor(
                topk_indices, local_topk_indices
            )

        return topk_indices

class IndexerBf16(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        index_k_norm_type: str,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_rope_neox_style = False,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if is_cuda():
            self.sm_count = deep_gemm_oss.get_num_sms()
            self.half_device_sm_count = align(self.sm_count // 2, 8)

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        if index_k_norm_type == "rms":
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.k_norm = V32LayerNorm(self.head_dim)
        # NOTE: weight_proj is not quantized
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=is_rope_neox_style,
            device="cuda" if not is_npu() else "npu",
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5
        self.topk = 2048

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,   # [l, h, d]
        weights: torch.Tensor, # [l, h, 1]
        key: torch.Tensor,     # [l, d]
        metadata: BaseIndexerMetadata,
    ):
        assert query.shape[0] == key.shape[0]
        bs = forward_batch.batch_size
        block_tables = metadata.get_page_table_64()
        page_size = 64
        max_seq_len = block_tables.shape[1] * page_size
        all_qk_logits = torch.full((query.shape[0], max_seq_len), float("-inf"), dtype=torch.float32, device=query.device)
        for i in range(bs):
            q_st = metadata.attn_metadata.cu_seqlens_q[i]
            q_ed = metadata.attn_metadata.cu_seqlens_q[i+1]
            q = query[q_st:q_ed]
            k = key[q_st:q_ed]
            w = weights[q_st:q_ed]

            l = q.shape[0]
            # [l, d] -> [h, l, d]
            k = k.unsqueeze(0).repeat_interleave(self.n_heads, 0)
            # [h, l, d] @ [h, d, l] -> [h, l, l]
            index_score = q.transpose(0, 1) @ k.transpose(-2, -1)
            # [h, l, l] -> [l, l]
            index_score = (w.transpose(0, 1) * F.relu(index_score)).sum(0)
            causal_mask = torch.triu(torch.full((l, l), float("-inf"), device=query.device), diagonal=1)
            index_score = index_score + causal_mask
            all_qk_logits[q_st:q_ed, :index_score.shape[1]] = index_score

        return all_qk_logits
    
    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,   # [b, h, d]
        weights: torch.Tensor, # [b, h, 1]
        metadata: BaseIndexerMetadata,
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)
        page_size = 64
        block_tables = metadata.get_page_table_64()
        seqlens_32 = metadata.get_seqlens_int32()
        index_k_cache = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )
        index_k_cache = index_k_cache.view(
            index_k_cache.shape[0], page_size, -1
        )
        max_seq_len = block_tables.shape[1] * page_size
        logits = triton_mqa_logits(query, weights.squeeze(2), index_k_cache, block_tables, seqlens_32, max_seq_len)
        return logits

    def _get_topk_paged_extend(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,   # [s, h, d]
        weights: torch.Tensor, # [s, h, 1]
        metadata: BaseIndexerMetadata,
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)
        page_size = 64
        block_tables = metadata.get_page_table_64()
        seqlens_32 = metadata.get_seqlens_int32()
        seq_lens_expanded = metadata.get_seqlens_expanded()
        index_k_cache = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )
        index_k_cache = index_k_cache.view(
            index_k_cache.shape[0], page_size, -1
        )
        bs = forward_batch.batch_size
        token_to_batch = torch.arange(bs, dtype=torch.int32, device=query.device).repeat_interleave(forward_batch.extend_seq_lens)
        max_seq_len = block_tables.shape[1] * page_size
        weights = weights.squeeze(2)
        logits = triton_mqa_extend_logits(
            query, weights, index_k_cache, block_tables, token_to_batch, seqlens_32, seq_lens_expanded, max_seq_len
        )
        return logits

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        unit_test: bool = False
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)
            assert isinstance(forward_batch.attn_backend, DpskSparseAttnBackend)

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )

        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        if not NSA_USE_REAL_INDEXER:  # temporary
            return self._forward_fake(x, q_lora, positions, forward_batch, layer_id)

        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

        q_rope, _ = torch.split(
            query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        query[..., : self.rope_head_dim] = q_rope
        key[..., : self.rope_head_dim] = k_rope

        forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(
            layer_id=layer_id,
            loc=forward_batch.out_cache_loc,
            index_k=key,
        )

        weights, _ = self.weights_proj(x.float())
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * self.softmax_scale

        if is_cuda():
            if forward_batch.forward_mode.is_decode_or_idle():
                qk_logits = self._get_topk_paged(
                    forward_batch, layer_id, query, weights, metadata
                )
            else:
                assert forward_batch.seq_lens_cpu is not None
                if torch.all(forward_batch.extend_prefix_lens == 0):
                    qk_logits = self._get_topk_ragged(
                        forward_batch, layer_id, query, weights, key, metadata
                    )
                else:
                    qk_logits = self._get_topk_paged_extend(
                        forward_batch, layer_id, query, weights, metadata
                    )
        topk_result = metadata.topk_transform(qk_logits, self.index_topk)
        if unit_test:
            return qk_logits, topk_result
        return topk_result

def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    assert seqlens.dtype == torch.int32 and seqlens.is_cuda
    return torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )

import triton
import triton.language as tl

@triton.jit
def triton_mqa_logits_kernel(
    q_ptr, # [b, h, d]
    w_ptr, # [b, h]
    k_cache_ptr, # [l, 64, d]
    r_ptr,
    block_table,
    seq_lens,
    q_stride_b, q_stride_h, q_stride_d,
    w_stride_b, w_stride_h,
    k_c_stride_l, k_c_stride_p, k_c_stride_d,
    r_stride_b, r_stride_l,
    block_table_b, block_table_d,
    HEAD_DIM: tl.constexpr = 128,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64
):
    pid_b = tl.program_id(0)
    cur_seq_len = tl.load(seq_lens + pid_b)
    block_table += pid_b * block_table_b
    r_ptr += pid_b * r_stride_b
    off_m = tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, HEAD_DIM)
    # [BM, d]
    q = tl.load(
        q_ptr + pid_b * q_stride_b +\
        off_m[:,None]*q_stride_h + off_d[None,:]
    )
    # [BM]
    w = tl.load(
        w_ptr + pid_b * w_stride_b + off_m
    )
    
    for i in range(0, tl.cdiv(cur_seq_len, BLOCK_N)):
        page_id = tl.load(block_table+i)
        k_st = i * BLOCK_N
        off_n = tl.arange(0, BLOCK_N)
        mask = (off_n + k_st) < cur_seq_len
        # [BN, d]
        k = tl.load(
            k_cache_ptr + page_id * k_c_stride_l +\
            off_n[:,None]*k_c_stride_p + off_d[None,:],
            mask=mask[:,None], other=0
        )
        # [BM, BN]
        s = tl.dot(q, k.trans())
        s = tl.maximum(s, 0.0)
        # [BN]
        r = tl.sum(s * w[:,None], axis=0)
        tl.store(r_ptr + k_st+off_n, r)


def triton_mqa_logits(
    query: torch.Tensor, # [b, h, d]
    weights, # [b, h]
    index_k_cache,
    block_tables, # [b, s]
    seqlens_32,
    max_seq_len,
):
    b, h, d = query.shape
    ret = query.new_empty(b, max_seq_len, dtype=torch.float32)
    triton_mqa_logits_kernel[(b,)](
        query,
        weights,
        index_k_cache,
        ret,
        block_tables,
        seqlens_32,
        *query.stride(),
        *weights.stride(),
        *index_k_cache.stride(),
        max_seq_len, 1,
        *block_tables.stride(),
        BLOCK_M=h
    )
    return ret

@triton.jit
def triton_mqa_extend_logits_kernel(
    q_ptr, # [s, h, d]
    w_ptr, # [s, h]
    k_cache_ptr, # [l, 64, d]
    r_ptr,
    block_table,
    token_to_batch, # [s]
    seq_lens, # [b]
    seq_lens_expanded, # [s]
    bs,
    q_stride_s, q_stride_h, q_stride_d,
    w_stride_s, w_stride_h,
    k_c_stride_l, k_c_stride_p, k_c_stride_d,
    r_stride_s, r_stride_l,
    block_table_b, block_table_d,
    HEAD_DIM: tl.constexpr = 128,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64
):
    # One CTA per q token
    pid = tl.program_id(0)
    pid_b = tl.load(token_to_batch + pid)
    cur_seq_len = tl.load(seq_lens_expanded + pid)
    block_table += pid_b * block_table_b
    r_ptr += pid * r_stride_s
    off_m = tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, HEAD_DIM)
    # [BM, d]
    q = tl.load(
        q_ptr + pid * q_stride_s + \
        off_m[:,None]*q_stride_h + off_d[None,:]
    )
    # [BM]
    w = tl.load(
        w_ptr + pid * w_stride_s + off_m
    )
    
    for i in range(0, tl.cdiv(cur_seq_len, BLOCK_N)):
        page_id = tl.load(block_table+i)
        k_st = i * BLOCK_N
        off_n = tl.arange(0, BLOCK_N)
        mask = (off_n + k_st) < cur_seq_len
        # [BN, d]
        k = tl.load(
            k_cache_ptr + page_id * k_c_stride_l + \
            off_n[:,None]*k_c_stride_p + off_d[None,:],
            mask=mask[:,None], other=0
        )
        # [BM, BN]
        s = tl.dot(q, k.trans())
        # relu
        s = tl.maximum(s, 0.0)
        # [BN]
        r = tl.sum(s * w[:,None], axis=0)
        tl.store(r_ptr + k_st+off_n, r)


def triton_mqa_extend_logits(
    query: torch.Tensor,    # [s, h, d]
    weights,                # [s, h]
    index_k_cache,          # [l, 64, d]
    block_tables,           # [b, s]
    token_to_batch,         # [s]
    seqlens_32,
    seq_lens_expanded,      # [s]
    max_seq_len,
):
    s, h, d = query.shape
    bs = block_tables.shape[0]
    ret = query.new_empty(s, max_seq_len, dtype=torch.float32)
    triton_mqa_extend_logits_kernel[(s,)](
        query,
        weights,
        index_k_cache,
        ret,
        block_tables,
        token_to_batch,
        seqlens_32,
        seq_lens_expanded,
        bs,
        *query.stride(),
        *weights.stride(),
        *index_k_cache.stride(),
        *ret.stride(),
        *block_tables.stride(),
        BLOCK_M=h
    )
    return ret
