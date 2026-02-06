import torch
from torch.nn import Linear, RMSNorm
import torch.nn.functional as F
import random
import types
from typing import Optional, List, Union
from einops import rearrange, repeat
from dataclasses import dataclass

from sglang.srt.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.layers.attention.dsa.nsa_indexer import IndexerBf16
from sglang.srt.layers.attention.dsa_backend import DpskSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool, ReqToTokenPool,
)

@dataclass
class ModelConfigMock:
    num_attention_heads = 32
    index_n_heads = 16
    index_head_dim = 128
    index_topk = 2048
    hidden_size = 3072
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    q_lora_rank = 1536
    layernorm_epsilon = 1e-6
    rope_theta = 10000000.0
    rope_scaling = None
    # {
    #     "beta_fast": 32,
    #     "beta_slow": 1,
    #     "factor": 40,
    #     "mscale": 1.0,
    #     "mscale_all_dim": 1.0,
    #     "original_max_position_embeddings": 4096,
    #     "type": "deepseek_yarn"
    # }
    max_position_embeddings = 133120 # 130k
    context_len = max_position_embeddings
    hf_config = types.SimpleNamespace()


model_config = ModelConfigMock()
page_size = 64

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, cos, sin, apply_rope_fusion=False, mla = False):
    if mla:
        b, s, h, d = t.shape
        t = t.view(b, s, h, d // 2, 2).transpose(4, 3).reshape(b, s, h, d)
    return (t * cos) + (rotate_half(t) * sin)

class RefIndexer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = model_config.index_n_heads
        self.head_dim = model_config.index_head_dim
        self.index_topk = model_config.index_topk
        
        self.dim = model_config.hidden_size
        self.rope_head_dim = model_config.qk_rope_head_dim
        self.q_lora_rank = model_config.q_lora_rank

        device = torch.cuda.current_device()
        dtype = torch.bfloat16

        self.wq_b = Linear(
            self.dim if self.q_lora_rank is None else self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False, device=device, dtype=dtype
        )
        self.wk = Linear(self.dim, self.head_dim, bias=False, device=device, dtype=dtype)
        self.weights_proj = Linear(self.dim, self.n_heads, bias=False, device=device, dtype=dtype)

        self.k_norm = RMSNorm(
            self.head_dim,
            eps=model_config.layernorm_epsilon,
            device=device,
        )

        self.softmax_scale = self.head_dim ** -0.5        
        self.apply_rope_fusion = False


    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        bsz, seqlen, _ = x.size()        
        q = self.wq_b(qr)
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
        # k = torch.matmul(x, self.wk.t())
        k = self.wk(x)
        k = self.k_norm(k).unsqueeze(-2)
        
        q = q.transpose(1,2).float()
        k = k.transpose(1,2).float()
        
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        
        q_pe = rearrange(q_pe, 'b h s d -> b s h d').contiguous()
        k_pe = rearrange(k_pe, 'b h s d -> b s h d').contiguous()
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, apply_rope_fusion = self.apply_rope_fusion, mla=True)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, apply_rope_fusion = self.apply_rope_fusion, mla=True)
        q_pe = rearrange(q_pe, 'b s h d -> b h s d').contiguous()
        k_pe = rearrange(k_pe, 'b s h d -> b h s d').contiguous()
        
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        # weights = torch.matmul(x, self.weights_proj.t()) * self.n_heads ** -0.5
        weights = self.weights_proj(x) * self.n_heads ** -0.5
        weights = weights.unsqueeze(-1) * self.softmax_scale
        weights = weights.transpose(1,2) # [bsz, n_heads, seq_len, 1]

        weights = weights.float()
        k = repeat(k, 'b 1 s d -> b h s d', h=q.size(1))
            
        index_score = q @ k.transpose(-2, -1)
        index_score = (weights * F.relu(index_score)).sum(1, keepdim=True) # (bsz, 1, seq_len, seq_len)
        index_score = index_score + mask
        
        topk = min(self.index_topk, seqlen)
        
        # if not self.dense_warmup:
        topk_indices = index_score.topk(topk, dim=-1)[1]
        # mask out the future tokens using seqlen as placeholder
        future_mask = torch.arange(seqlen, device=x.device)[None, None, :, None] < torch.arange(topk, device=x.device)[None, None, None, :]
        topk_indices.masked_fill_(future_mask, seqlen)
            
        index_mask = torch.full((bsz, 1, seqlen+1, seqlen+1), torch.finfo(x.dtype).min, device=x.device).scatter_(-1, topk_indices, 0)
        index_mask = index_mask[:, :, :seqlen, :seqlen]
        index_score = index_score + index_mask

        index_score = index_score.clip(min=torch.finfo(x.dtype).min)

        return topk_indices, index_score, index_mask

def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_forward_batch(
    seq_lens: List[int], 
    extend_prefix_lens: List[int],
) -> ForwardBatch:
    batch_size = len(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    seq_lens_sum = seq_lens.sum().item()
    extend_prefix_lens = torch.tensor(extend_prefix_lens, dtype=torch.int32)
    extend_prefix_lens_cpu = extend_prefix_lens.tolist()
    # s_q
    extend_seq_lens = seq_lens - extend_prefix_lens
    extend_seq_lens_cpu = extend_seq_lens.tolist()
    extend_num_tokens = extend_seq_lens.sum().item()
    total_token_num = extend_num_tokens
    positions = torch.tensor(
        [
            position
            for extend_prefix_len, seq_len in zip(extend_prefix_lens, seq_lens)
            for position in range(extend_prefix_len.item(), seq_len.item())
        ],
        dtype=torch.int32
    )

    if all(extend_seq_len == 1 for extend_seq_len in extend_seq_lens_cpu):
        forward_mode = ForwardMode.DECODE
    else:
        forward_mode = ForwardMode.EXTEND

    req_to_token_pool = ReqToTokenPool(
        size=batch_size,
        max_context_len=model_config.max_position_embeddings + page_size,
        device=torch.get_default_device(),
        enable_memory_saver=False,
    )
    req_pool_indices = torch.empty((batch_size,), dtype=torch.int32)
    out_cache_loc = torch.empty((total_token_num,), dtype=torch.int32)

    allocated_tokens = 0
    allocated_new_tokens = 0
    for idx, seq_len in enumerate(seq_lens):
       s_q = extend_seq_lens[idx]
       req_pool_indices[idx] = idx
       req_to_token_pool.req_to_token[idx][:seq_len] = torch.tensor(range(seq_len), dtype=torch.int32) + allocated_tokens
       out_cache_loc[allocated_new_tokens: allocated_new_tokens + s_q] = req_to_token_pool.req_to_token[idx][seq_len-s_q:seq_len]
       # align to next page
       allocated_tokens += seq_len + page_size - seq_len % page_size
       allocated_new_tokens += s_q

    token_to_kv_pool = DSATokenToKVPool(
        size=seq_lens_sum + batch_size * page_size,
        model_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        kv_lora_rank=model_config.kv_lora_rank,
        qk_rope_head_dim=model_config.qk_rope_head_dim,
        layer_num=1,
        device=torch.get_default_device(),
        enable_memory_saver=False,
        max_batch_size=batch_size,
        max_context_len=model_config.max_position_embeddings + page_size,
        page_size=page_size,
        rank=0,
        index_head_dim=model_config.index_head_dim,
        index_dtype=torch.bfloat16,
    )
   
    forward_batch = ForwardBatch(
        batch_size=batch_size,
        input_ids=None,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_sum=seq_lens_sum,
        seq_lens_cpu=seq_lens.cpu(),
        forward_mode=forward_mode,
        positions=positions,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        out_cache_loc=out_cache_loc,
        extend_num_tokens = extend_num_tokens,
        extend_seq_lens = extend_seq_lens,
        extend_seq_lens_cpu=extend_seq_lens.cpu(),
        extend_prefix_lens = extend_prefix_lens,
        extend_prefix_lens_cpu = extend_prefix_lens_cpu,
    )
    forward_batch.tp_num_tokens = total_token_num

    # attn backend metadata
    model_runner_mock = types.SimpleNamespace()
    model_runner_mock.is_draft_worker = False
    model_runner_mock.device = torch.cuda.current_device()
    model_runner_mock.page_size = page_size
    model_runner_mock.token_to_kv_pool = token_to_kv_pool
    model_runner_mock.model_config = model_config
    model_runner_mock.model_config.hf_config.architectures = ["FLASHForCausalLM"]
    model_runner_mock.model_config.hf_config.index_topk = model_config.index_topk
    model_runner_mock.req_to_token_pool = req_to_token_pool
    model_runner_mock.server_args = types.SimpleNamespace()
    model_runner_mock.server_args.speculative_num_draft_tokens = 1

    forward_batch.attn_backend = DpskSparseAttnBackend(model_runner_mock)
    return forward_batch

def _compute_inv_freq() -> torch.Tensor:
    """Compute the inverse frequency."""
    # NOTE(woosuk): To exactly match the HF implementation, we need to
    # use CPU to compute the cache and then move it to GPU. However, we
    # create the cache on GPU for faster initialization. This may cause
    # a slight numerical difference between the HF implementation and ours.
    inv_freq = 1.0 / (
        model_config.rope_theta
        ** (
            torch.arange(0, model_config.qk_rope_head_dim, 2, dtype=torch.float) / model_config.qk_rope_head_dim
        )
    )
    return inv_freq

def _compute_cos_sin_cache() -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = _compute_inv_freq()
    t = torch.arange(model_config.max_position_embeddings, dtype=torch.float)

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin

def main():
    set_all_seeds(22)
    device = torch.cuda.current_device()
    torch.set_default_device(device)

    ref_indexer = RefIndexer()
    c = model_config
    with set_default_torch_dtype(torch.bfloat16):
        srt_indexer = IndexerBf16(
            c.hidden_size, c.index_n_heads, c.index_head_dim, c.qk_rope_head_dim,
            c.index_topk, c.q_lora_rank, c.max_position_embeddings,
            c.rope_theta, 0,
            "", 128, # use for fp8
            c.rope_scaling,
            False
        )
    srt_indexer.wq_b.weight.copy_(ref_indexer.wq_b.weight)
    srt_indexer.wk.weight.copy_(ref_indexer.wk.weight)
    srt_indexer.weights_proj.weight.copy_(ref_indexer.weights_proj.weight)
    srt_indexer.k_norm.weight.data.copy_(ref_indexer.k_norm.weight)

    dtype = torch.bfloat16
    b, s = 4, 1024
    qr = torch.randn(b, s, model_config.q_lora_rank, dtype=dtype)
    x = torch.randn(b, s, model_config.hidden_size, dtype=dtype)
    cos, sin = _compute_cos_sin_cache()
    cos = cos[:s]
    sin = sin[:s]
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    cos = cos[None, :, None]
    sin = sin[None, :, None]
    print(f"{cos.shape=} {sin.shape=}")
    mask = torch.full((s, s), float("-inf"), dtype=torch.float32).triu(1)
    mask = mask[None, None]
    _, ref_qk_logits, _ = ref_indexer(x, qr, cos, sin, mask)

    seq_lens = [1024] * 4
    extend_prefix_lens = [0 for s in seq_lens]
    forward_batch = prepare_forward_batch(seq_lens, extend_prefix_lens)
    # print(f"{forward_batch=}")
    assert isinstance(forward_batch.attn_backend, DpskSparseAttnBackend)
    forward_batch.attn_backend.init_forward_metadata(
        forward_batch
    )
    # print(f"forward_metadata = {forward_batch.attn_backend.forward_metadata}")
    x = x.view(-1, c.hidden_size)
    qr = qr.view(-1, c.q_lora_rank)
    srt_qk_logits, _ = srt_indexer.forward(x, qr, forward_batch.positions, forward_batch, 0, True)

    print(f"{srt_qk_logits.shape=}")
    print(f"{ref_qk_logits.shape=}")
    ref_qk_logits = ref_qk_logits.view(srt_qk_logits.shape[0], -1)
    calc_cos = lambda a, b: 1 - 2 * (a * b).sum().item() / max((a * a + b * b).sum().item(), 1e-12)
    srt_qk_logits[srt_qk_logits<-1e10] = 0
    ref_qk_logits[ref_qk_logits<-1e10] = 0
    cos_err = calc_cos(srt_qk_logits, ref_qk_logits)
    print(f"{cos_err=}")
    assert cos_err < 5e-5


if __name__ == "__main__":
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://localhost:{random.randint(10000, 20000)}",
    )
    initialize_model_parallel()
    initialize_dp_attention(
        attn_tp_rank=0,
        attn_tp_size=1,
        dp_size=1,
        dp_rank=0,
        global_rank=0,
        local_rank=0,
        hidden_size=1,
        max_num_tokens=2048,
        force_deterministic_rsag=False
    )
    try:
        main()
        print(f"Success!")
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
