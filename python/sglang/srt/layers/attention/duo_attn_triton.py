import math
import torch
import triton
import triton.language as tl


@triton.jit
def streaming_attention_with_kvcache_kernel(
    Q, K_cache, V_cache,
    CacheSeqlens, 
    Out,
    Meta,
    StreamingInfo, # [num_heads, 2]
    HeadMaskType, # [num_heads]
    BlockTables,
    softmax_scale,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_mb, stride_mm, stride_mh, stride_md,
    stride_btb, stride_bts,
    stride_sinfo_h, stride_sinfo_d, # streaming_info strides
    stride_hmask, # head_mask_type stride
    BLOCK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_PER_SPLIT: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_PAGED_KV: tl.constexpr,
    NUM_QUERY_GROUPS: tl.constexpr,
    SPLIT_K: tl.constexpr,
):

    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_h_kv = off_h // NUM_QUERY_GROUPS
    off_split = tl.program_id(2)
    
    # Load sequence length for this batch.
    cache_seqlen = tl.load(CacheSeqlens + off_b)

    if cache_seqlen == 0:
        return
    
    start_split = off_split * BLOCK_N_PER_SPLIT
    end_split = min((off_split + 1) * BLOCK_N_PER_SPLIT, cache_seqlen)
    
    # TODO: Is it possible to avoid redundant kernel launching?
    if start_split >= cache_seqlen:
        return
    
    # Load head mask type (-1 for streaming, 0 for full).
    head_mask_type = tl.load(HeadMaskType + off_h * stride_hmask)

    # Load sink and recent block counts, and determine boundary.
    sink_blocks = tl.load(StreamingInfo + off_h * stride_sinfo_h + 0 * stride_sinfo_d)
    recent_blocks = tl.load(StreamingInfo + off_h * stride_sinfo_h + 1 * stride_sinfo_d)
    total_blocks = (cache_seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE  # Ceiling division.

    last_block_tokens = cache_seqlen % BLOCK_SIZE
    sink_tokens = sink_blocks * BLOCK_SIZE
    if last_block_tokens > 0:
        recent_tokens = (recent_blocks - 1) * BLOCK_SIZE + last_block_tokens
    else:
        recent_tokens = recent_blocks * BLOCK_SIZE

    start_sink = 0
    end_sink = min(sink_tokens, cache_seqlen)

    start_recency = max(cache_seqlen - recent_tokens, end_sink)
    end_recency = cache_seqlen
    
    offs_m = tl.arange(0, BLOCK_M) # Assume query length is 1, then not used.
    offs_n = tl.arange(0, BLOCK_N)
    offs_p = tl.arange(0, BLOCK_P)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load query.
    q_ptrs = Q + off_b * stride_qb + \
                off_h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs)
    
    # Initialize online softmax states.
    m_i = - float("inf")
    l_i = - float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    if USE_PAGED_KV: # Paged KV cache.
        if head_mask_type == -1: # Streaming attention head.
            # Have the intersection of the current split and the sink.
            start_split_sink = max(start_split, start_sink)
            end_split_sink = min(end_split, end_sink)
            # Process sink blocks.
            for start_n in range(start_split_sink, end_split_sink, BLOCK_P):
                logical_block_idx = start_n // BLOCK_P
                physical_block_idx = tl.load(BlockTables + off_b * stride_btb + logical_block_idx * stride_bts)
                
                # Load paged key.
                k_ptrs = K_cache + physical_block_idx * stride_kb + offs_p[:, None] * stride_kt + \
                    off_h_kv * stride_kh + offs_d[None, :] * stride_kd
                k = tl.load(k_ptrs, mask=start_n + offs_p[:, None] < end_split_sink, other=0.)
                
                # Compute attention scores.
                qk = tl.zeros([BLOCK_P], dtype=tl.float32)
                qk += tl.sum(q[None, :] * k, 1) * softmax_scale
                
                # Mask for valid keys.
                mask_k_col = start_n + offs_p < end_split_sink
                mask_attn = mask_k_col
                qk += tl.where(mask_attn, 0, float("-inf"))
                
                # Online softmax update.
                m_ij = tl.maximum(tl.max(qk, 0), m_i)
                p = tl.exp(qk - m_ij)
                l_ij = tl.sum(p, 0)
                
                # Load paged value.
                v_ptrs = V_cache + physical_block_idx * stride_vb + offs_p[:, None] * stride_vt + \
                    off_h_kv * stride_vh + offs_d[None, :] * stride_vd
                v = tl.load(v_ptrs, mask=start_n + offs_p[:, None] < end_split_sink, other=0.)
                
                # Update accumulator.
                acc_scale = tl.exp(m_i - m_ij)
                acc = acc * acc_scale
                p = p.to(v.dtype)
                acc += tl.sum(p[:, None] * v, 0)
                
                m_i = m_ij
                l_i_new = tl.exp(l_i - m_ij) + l_ij
                l_i = m_ij + tl.log(l_i_new)
            
            # Have the intersection of the current split and the recency.
            start_split_recency = max(start_split, start_recency)
            end_split_recency = min(end_split, end_recency)
            # Process recent blocks.
            for start_n in range(start_split_recency, end_split_recency, BLOCK_P):
                logical_block_idx = start_n // BLOCK_P
                physical_block_idx = tl.load(BlockTables + off_b * stride_btb + logical_block_idx * stride_bts)
                
                # Load paged key.
                k_ptrs = K_cache + physical_block_idx * stride_kb + offs_p[:, None] * stride_kt + \
                    off_h_kv * stride_kh + offs_d[None, :] * stride_kd
                k = tl.load(k_ptrs, mask=start_n + offs_p[:, None] < end_split_recency, other=0.0)
                
                # Compute attention scores.
                qk = tl.zeros([BLOCK_P], dtype=tl.float32)
                qk += tl.sum(q[None, :] * k, 1) * softmax_scale
                
                # Mask for valid keys.
                mask_k_col = start_n + offs_p < end_split_recency
                mask_attn = mask_k_col
                qk += tl.where(mask_attn, 0, float("-inf"))
                
                # Online softmax update.
                m_ij = tl.maximum(tl.max(qk, 0), m_i)
                p = tl.exp(qk - m_ij)
                l_ij = tl.sum(p, 0)
                
                # Load paged value.
                v_ptrs = V_cache + physical_block_idx * stride_vb + offs_p[:, None] * stride_vt + \
                    off_h_kv * stride_vh + offs_d[None, :] * stride_vd
                v = tl.load(v_ptrs, mask=start_n + offs_p[:, None] < end_split_recency, other=0.0)
                
                # Update accumulator.
                acc_scale = tl.exp(m_i - m_ij)
                acc = acc * acc_scale
                p = p.to(v.dtype)
                acc += tl.sum(p[:, None] * v, 0)
                
                m_i = m_ij
                l_i_new = tl.exp(l_i - m_ij) + l_ij
                l_i = m_ij + tl.log(l_i_new)
        else: # Full attention head.
            for start_n in range(start_split, end_split, BLOCK_P):
                logical_block_idx = start_n // BLOCK_P
                physical_block_idx = tl.load(BlockTables + off_b * stride_btb + logical_block_idx * stride_bts)
                
                # Load key.
                k_ptrs = K_cache + physical_block_idx * stride_kb + offs_p[:, None] * stride_kt + \
                    off_h_kv * stride_kh + offs_d[None, :] * stride_kd
                k = tl.load(k_ptrs, mask=start_n + offs_p[:, None] < end_split, other=0.0)
                
                # Compute attention scores.
                qk = tl.zeros([BLOCK_P], dtype=tl.float32)
                qk += tl.sum(q[None, :] * k, 1) * softmax_scale
                
                # Mask for valid keys.
                mask_k_col = start_n + offs_p < end_split
                mask_attn = mask_k_col
                qk += tl.where(mask_attn, 0, float("-inf"))
                
                # Online softmax update.
                m_ij = tl.maximum(tl.max(qk, 0), m_i)
                p = tl.exp(qk - m_ij)
                l_ij = tl.sum(p, 0)
                
                # Load paged value.
                v_ptrs = V_cache + physical_block_idx * stride_vb + offs_p[:, None] * stride_vt + \
                    off_h_kv * stride_vh + offs_d[None, :] * stride_vd
                v = tl.load(v_ptrs, mask=start_n + offs_p[:, None] < end_split, other=0.0)
                
                # Update accumulator
                acc_scale = tl.exp(m_i - m_ij)
                acc = acc * acc_scale
                p = p.to(v.dtype)
                acc += tl.sum(p[:, None] * v, 0)
                
                m_i = m_ij
                l_i_new = tl.exp(l_i - m_ij) + l_ij
                l_i = m_ij + tl.log(l_i_new)
    else: # Contiguous KV cache.
        if head_mask_type == -1:  # Streaming attention head.
            # Have the intersection of the current split and the sink.
            start_split_sink = max(start_split, start_sink)
            end_split_sink = min(end_split, end_sink)
            # Process sink tokens.
            for start_n in range(start_split_sink, end_split_sink, BLOCK_N):
                # Load key.
                k_ptrs = K_cache + off_b * stride_kb + (start_n + offs_n[:, None]) * stride_kt + \
                    off_h_kv * stride_kh + offs_d[None, :] * stride_kd
                k = tl.load(k_ptrs, mask=start_n + offs_n[:, None] < end_split_sink, other=0.0)
                
                # Compute attention scores.
                qk = tl.zeros([BLOCK_N], dtype=tl.float32)
                qk += tl.sum(q[None, :] * k, 1) * softmax_scale
                
                # Mask for valid keys.
                mask_k_col = start_n + offs_n < end_split_sink
                mask_attn = mask_k_col
                qk += tl.where(mask_attn, 0, float("-inf"))
                
                # Online softmax update.
                m_ij = tl.maximum(tl.max(qk, 0), m_i)
                p = tl.exp(qk - m_ij)
                l_ij = tl.sum(p, 0)
                
                # Load value.
                v_ptrs = V_cache + off_b * stride_vb + (start_n + offs_n[:, None]) * stride_vt + \
                    off_h_kv * stride_vh + offs_d[None, :] * stride_vd
                v = tl.load(v_ptrs, mask=start_n + offs_n[:, None] < end_split_sink, other=0.0)
                
                # Update accumulator.
                acc_scale = tl.exp(m_i - m_ij)
                acc = acc * acc_scale
                p = p.to(v.dtype)
                acc += tl.sum(p[:, None] * v, 0)
                
                m_i = m_ij
                l_i_new = tl.exp(l_i - m_ij) + l_ij
                l_i = m_ij + tl.log(l_i_new)
            
            # Have the intersection of the current split and the recency.
            start_split_recency = max(start_split, start_recency)
            end_split_recency = min(end_split, end_recency)
            # Process recent tokens.
            for start_n in range(start_split_recency, end_split_recency, BLOCK_N):
                # Load key.
                k_ptrs = K_cache + off_b * stride_kb + (start_n + offs_n[:, None]) * stride_kt + \
                   off_h_kv * stride_kh + offs_d[None, :] * stride_kd
                k = tl.load(k_ptrs, mask=start_n + offs_n[:, None] < end_split_recency, other=0.0)
                
                # Compute attention scores.
                qk = tl.zeros([BLOCK_N], dtype=tl.float32)
                qk += tl.sum(q[None, :] * k, 1) * softmax_scale
                
                # Mask for valid keys.
                mask_k_col = start_n + offs_n < end_split_recency
                mask_attn = mask_k_col
                qk += tl.where(mask_attn, 0, float("-inf"))
                
                # Online softmax update.
                m_ij = tl.maximum(tl.max(qk, 0), m_i)
                p = tl.exp(qk - m_ij)
                l_ij = tl.sum(p, 0)
                
                # Load value block.
                v_ptrs = V_cache + off_b * stride_vb + (start_n + offs_n[:, None]) * stride_vt + \
                    off_h_kv * stride_vh + offs_d[None, :] * stride_vd
                v = tl.load(v_ptrs, mask=start_n + offs_n[:, None] < end_split_recency, other=0.0)
                
                # Update accumulator.
                acc_scale = tl.exp(m_i - m_ij)
                acc = acc * acc_scale
                p = p.to(v.dtype)
                acc += tl.sum(p[:, None] * v, 0)
                
                m_i = m_ij
                l_i_new = tl.exp(l_i - m_ij) + l_ij
                l_i = m_ij + tl.log(l_i_new)
        else: # Full attention head.
            for start_n in range(start_split, end_split, BLOCK_N):
                # Load key.
                k_ptrs = K_cache + off_b * stride_kb + (start_n + offs_n[:, None]) * stride_kt + \
                    off_h_kv * stride_kh + offs_d[None, :] * stride_kd
                k = tl.load(k_ptrs, mask=start_n + offs_n[:, None] < end_split, other=0.0)
                
                # Compute attention scores.
                qk = tl.zeros([BLOCK_N], dtype=tl.float32)
                qk += tl.sum(q[None, :] * k, 1) * softmax_scale
                
                # Mask for valid keys.
                mask_k_col = start_n + offs_n < end_split
                mask_attn = mask_k_col
                qk += tl.where(mask_attn, 0, float("-inf"))
                
                # Online softmax update.
                m_ij = tl.maximum(tl.max(qk, 0), m_i)
                p = tl.exp(qk - m_ij)
                l_ij = tl.sum(p, 0)
                
                # Load value.
                v_ptrs = V_cache + off_b * stride_vb + (start_n + offs_n[:, None]) * stride_vt + \
                    off_h_kv * stride_vh + offs_d[None, :] * stride_vd
                v = tl.load(v_ptrs, mask=start_n + offs_n[:, None] < end_split, other=0.0)
                
                # Update accumulator.
                acc_scale = tl.exp(m_i - m_ij)
                acc = acc * acc_scale
                p = p.to(v.dtype)
                acc += tl.sum(p[:, None] * v, 0)
                
                m_i = m_ij
                l_i_new = tl.exp(l_i - m_ij) + l_ij
                l_i = m_ij + tl.log(l_i_new)
    
    if SPLIT_K > 1:
        # Store partial results for this split.
        # [acc, m_i, l_i].
        partial_out_ptrs = Out + off_b * stride_ob + \
                off_split * stride_om + off_h * stride_oh + offs_d * stride_od
        tl.store(partial_out_ptrs, acc)
        partial_meta_ptr = Meta + off_b * stride_mb + \
                off_split * stride_mm + off_h * stride_mh + 0 * stride_md
        tl.store(partial_meta_ptr, m_i)
        partial_meta_ptr = Meta + off_b * stride_mb + \
                off_split * stride_mm + off_h * stride_mh + 1 * stride_md
        tl.store(partial_meta_ptr, l_i)
    else:
        o_scale = tl.exp(m_i - l_i)
        acc = acc * o_scale

        out_ptrs = Out + off_b * stride_ob + \
                off_h * stride_oh + offs_d * stride_od
        tl.store(out_ptrs, acc)


def streaming_attention_with_kvcache(
    q, k_cache, v_cache, cache_seqlens, 
    out=None, 
    streaming_info=None, # [num_heads, 2]
    head_mask_type=None, # [num_heads]
    block_tables=None, 
    softmax_scale=None, 
    causal=True, 
    block_size=16,
):
    num_seqs, seq_len, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[-2] if k_cache.dim() == 4 else k_cache.shape[1] // block_size
    num_query_groups = num_heads // num_kv_heads
    
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    max_seqlen_k = cache_seqlens.max().item()
    
    if out is None:
        out = torch.empty_like(q)
    
    # Handle default streaming_info and head_mask_type.
    if streaming_info is None:
        streaming_info = torch.zeros((num_heads, 2), dtype=torch.int32, device=q.device)
    # NOTE: `block_sparse_attn` forces block size of 128 for implementation brevity.
    # See https://github.com/mit-han-lab/Block-Sparse-Attention/issues/6
    BLOCK_SIZE = 128
    MIN_BLOCK_N_PER_SPLIT = streaming_info.max().item() * 128
    if head_mask_type is None:
        head_mask_type = torch.zeros((num_heads,), dtype=torch.int32, device=q.device)
    
    if block_tables is None:
        use_paged_kv = False
        block_tables = torch.zeros((num_seqs, 1), dtype=torch.int32, device=q.device)
    else:
        use_paged_kv = True
    
    # Adjust block sizes.
    BLOCK_P = block_size
    BLOCK_M = 1 # Query length is 1 during decode.
    BLOCK_N = max(64, min(64, triton.next_power_of_2(max_seqlen_k)))
    BLOCK_N_PER_SPLIT = max(BLOCK_N, MIN_BLOCK_N_PER_SPLIT)
    BLOCK_DMODEL = head_dim
    assert BLOCK_N % BLOCK_P == 0, "Block size must be divisible by page size"
    assert BLOCK_SIZE % BLOCK_P == 0
    
    split_k = triton.cdiv(max_seqlen_k, BLOCK_N_PER_SPLIT)

    # Create temporary buffer for partial output if using split-k.
    if split_k > 1:
        # [num_seqs, split_k, num_heads, head_dim]
        partial_out = torch.zeros(
            (num_seqs, split_k, num_heads, head_dim),
            dtype=torch.float32, device=q.device
        )
        partial_meta = torch.ones(
            (num_seqs, split_k, num_heads, 2),
            dtype=torch.float32, device=q.device
        ) * (- float('inf'))
    else:
        # Dummy, won't be used.
        partial_out = out
        partial_meta = torch.ones(
            (num_seqs, 1, num_heads, 2),
            dtype=q.dtype, device=q.device
        ) * (- float('inf'))
    
    grid = (num_seqs, num_heads, split_k,)

    # Launch kernel.
    streaming_attention_with_kvcache_kernel[grid](
        q, k_cache, v_cache, 
        cache_seqlens, 
        partial_out,
        partial_meta,
        streaming_info,
        head_mask_type,
        block_tables,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_meta.stride(0), partial_meta.stride(1), partial_meta.stride(2), partial_meta.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        streaming_info.stride(0), streaming_info.stride(1),
        head_mask_type.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_P=BLOCK_P,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_N_PER_SPLIT=BLOCK_N_PER_SPLIT,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=causal,
        USE_PAGED_KV=use_paged_kv,
        NUM_QUERY_GROUPS=num_query_groups,
        SPLIT_K=split_k,
    )

    if split_k > 1:
        # NOTE: the actual split_k (determined by cache_seqlen) could be 
        # smaller than split_k (by max_seqlen_k).
        reduce_grid = (num_seqs, num_heads,)
        flash_attention_with_kvcache_reduce_kernel[reduce_grid](
            partial_out,
            partial_meta,
            out,
            partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
            partial_meta.stride(0), partial_meta.stride(1), partial_meta.stride(2), partial_meta.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_DMODEL=BLOCK_DMODEL,
            SPLIT_K=split_k,
        )
    
    return out

@triton.jit
def flash_attention_with_kvcache_reduce_kernel(
    PartialOut,
    PartialMeta,
    Out,
    stride_pb, stride_pm, stride_ph, stride_pd,
    stride_mb, stride_mm, stride_mh, stride_md,
    stride_ob, stride_om, stride_oh, stride_od,
    BLOCK_DMODEL: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize reduction state.
    m = - float("inf")
    l = - float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Load and reduce partial out.
    for split_idx in range(SPLIT_K):
        partial_out_ptrs = PartialOut + off_b * stride_pb + \
            split_idx * stride_pm + off_h * stride_ph + offs_d * stride_pd
        acc_i = tl.load(partial_out_ptrs)
        partial_meta_ptr = PartialMeta + off_b * stride_mb + \
            split_idx * stride_mm + off_h * stride_mh + 0 * stride_md
        m_i = tl.load(partial_meta_ptr)
        partial_meta_ptr = PartialMeta + off_b * stride_mb + \
            split_idx * stride_mm + off_h * stride_mh + 1 * stride_md
        l_i = tl.load(partial_meta_ptr)

        # Update global max.
        m_new = tl.maximum(m_i, m)
        
        # Scale existing states.
        acc_scale = tl.exp(m - m_new)
        acc = acc * acc_scale
        acc += acc_i * tl.exp(m_i - m_new)

        m = m_new
        l_scale = tl.exp(l - m_new)
        l_new = l_scale + tl.exp(l_i - m_new)
        l = m_new + tl.log(l_new)
     
    # Final normalization.
    o_scale = tl.exp(m - l)
    acc = acc * o_scale

    # Store final out.
    out_ptrs = Out + off_b * stride_ob + \
                off_h * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, acc)
