from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch_npu

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

class NpuMLAAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        v = torch.nn.functional.pad(v, [0, layer.qk_head_dim - layer.v_head_dim], value=0)

        B = len(forward_batch.seq_lens)
        N = q.shape[1]
        S = max(forward_batch.seq_lens)
        D = q.shape[2]

        batched_q = []; batched_k = []; batched_v = []
        start = 0
        for l in forward_batch.seq_lens:
            q_i = q[start:start+l]  # [l, N, D]
            k_i = k[start:start+l]  # [l, N, D]
            v_i = v[start:start+l]  # [l, N, D]
            if l < S: # pad to S
                pad_len = S - l
                q_i = torch.cat([q_i, torch.zeros(pad_len, N, D, dtype=q.dtype, device=q.device)], dim=0)
                k_i = torch.cat([k_i, torch.zeros(pad_len, N, D, dtype=q.dtype, device=q.device)], dim=0)
                v_i = torch.cat([v_i, torch.zeros(pad_len, N, D, dtype=q.dtype, device=q.device)], dim=0)
            batched_q.append(q_i)
            batched_k.append(k_i)
            batched_v.append(v_i)
            start += l

        if start != q.shape[0]:
            print(f"q.shape[0] unmatch seq_lens {start=} {q.shape[0]=}")
        assert start == q.shape[0], "q.shape[0] unmatch seq_lens, chunk prefill need support for mla"

        batched_q = torch.stack(batched_q, dim=0)  # [B, S, N, D]
        batched_k = torch.stack(batched_k, dim=0)  # [B, S, N, D]
        batched_v = torch.stack(batched_v, dim=0)  # [B, S, N, D]
        batched_q = batched_q.permute(0, 2, 1, 3)  # [B, N, S, D]
        batched_k = batched_k.permute(0, 2, 1, 3)  # [B, N, S, D]
        batched_v = batched_v.permute(0, 2, 1, 3)  # [B, N, S, D]

        attn_mask = ~torch.tril(torch.ones((1, 2048, 2048), dtype=torch.bool, device="npu"))

        o, _ = torch_npu.npu_fused_infer_attention_score(
            batched_q, batched_k, batched_v,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="BNSD",
            atten_mask=attn_mask,
            sparse_mode=2,
            scale=layer.scaling,
            actual_seq_lengths=forward_batch.seq_lens,
            actual_seq_lengths_kv=forward_batch.seq_lens)

        # o restore from BSND to SND
        pieces = []
        for i, l in enumerate(forward_batch.seq_lens):
            # Take the first l tokens of the i-th req
            pieces.append(o[i, :, :l, :].transpose(0, 1))  # [l, N, D]
        o = torch.cat(pieces, dim=0)  # [token_num, N, D]

        o = o.view(-1, layer.tp_q_head_num, layer.qk_head_dim)[..., :layer.v_head_dim]
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        # SND -> NSD -> BNSD
        q = q.movedim(0, 1).unsqueeze(0)
        k = k.movedim(0, 1).unsqueeze(0)
        v = v.movedim(0, 1).unsqueeze(0)

        key_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id) # shape [total_token_nums, 1/*kv_head_num*/, 576]
        value_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id) # shape [total_token_nums, 1/*kv_head_num*/, 512]

        # Adjust key_cache layout [block_nums, 1, block_size, head_dim]
        block_size = 128 # Must keep consistent with model_runner.py: self.page_size = 128
        key_cache = key_cache.view(-1, block_size, key_cache.shape[1], key_cache.shape[2]) # adjust to [block_nums, block_size, 1, 576]
        value_cache = value_cache.view(-1, block_size, value_cache.shape[1], value_cache.shape[2]) # adjust to [block_nums, block_size, 1, 512]
        key_cache = key_cache.movedim(1, 2) # adjust to [block_nums, 1, block_size, 576]
        value_cache = value_cache.movedim(1, 2) # adjust to [block_nums, 1, block_size, 512]

        k_pe = key_cache[..., 512:] # only take the last 64 dimensions

        # build block_table
        block_table = torch.full(
            size=(len(forward_batch.seq_lens), 2048),
            fill_value=-1,
            dtype=torch.int32,
            device="npu")

        for seq_idx in range(len(forward_batch.seq_lens)):
            req_pool_info = forward_batch.req_to_token_pool.get_req_pool_info(forward_batch.req_pool_indices[seq_idx].item())
            req_num_pages = (forward_batch.seq_lens[seq_idx] + block_size - 1) // block_size
            page_indices = req_pool_info.alloced_slots[::block_size] // block_size
            block_table[seq_idx, :req_num_pages] = page_indices[:req_num_pages]

        q_nope, q_pe = q.split([512, 64], dim=-1)
        k_nope = value_cache

        output, _ = torch_npu.npu_fused_infer_attention_score(
                q_nope, k_nope, value_cache,
                query_rope=q_pe,
                key_rope=k_pe,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=1,
                input_layout="BNSD",
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=block_table,
                block_size=block_size,
                actual_seq_lengths_kv=forward_batch.seq_lens)

        output = output.squeeze(0).movedim(1, 0) # BNSD -> SND
        return output
