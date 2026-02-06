# python3 test/srt/layers/native_sparse_attention/test_nsa_mla.py
import random
import unittest
import math
import torch
from torch import nn

from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.mem_cache.memory_pool import (
    MLATokenToKVPool, ReqToTokenPool,
    NativeSparseMLATokenToKVPool,
)
from dataclasses import dataclass
from typing import List

@dataclass
class AttnConfig:
    hidden_size: int = 2048
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    kv_lora_rank: int = 512
    scaling: float = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5
    num_attention_heads = 64
    num_key_value_heads = 64
    
    max_position_embeddings: int = 32768
    context_len: int = max_position_embeddings
    rope_theta: float = 50000.0
    rope_scaling = None
    rms_norm_eps: float = 1e-5
    
    attention_bias: bool = False
    attention_dropout: float = 0.0
    q_lora_rank = None
    
    kernel_size: int = 32
    stride: int = 16
    select_size: int = 64
    top_n: int = 20
    slc_att_num_init_blocks: int = 1
    slc_att_num_local_blocks: int = 7
    window_size: int = 512
    
    gate_proj_init_std: float = 0.02
    nsa_gate_mask: str = "110"
    head_not_share_att_gate_weight = True
    gate_proj_init_std = 0.006
    nsa_gate_type = "softmax"
    virtual_k_group_size = 16
    virtual_k_group_agg_type = "max"

    page_size: int = 64

def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HFModule(nn.Module):
    def __init__(self, config: AttnConfig):
        super().__init__()
        from hf_nsa_mla_v1_1 import DeepseekV3NSAWithMLA
        self.attn = DeepseekV3NSAWithMLA(config, layer_idx=0)

    def forward(            
        self,
        hidden_states: List[torch.Tensor],
    ):
        o = []
        latent_cache = []
        compressed_latent_cache = []
        for per_seq_hidden_state in hidden_states:
            per_seq_o, _, _, per_seq_latent_cache, per_seq_compressed_latent_cache = self.attn(per_seq_hidden_state.unsqueeze(0))
            o.append(per_seq_o.squeeze(0))
            latent_cache.append(per_seq_latent_cache.squeeze(0))
            compressed_latent_cache.append(per_seq_compressed_latent_cache.squeeze(0))
        return o, latent_cache, compressed_latent_cache
    
    def rand_weights(self, config: AttnConfig, seed: int):
        set_all_seeds(seed)
        for name, param in self.attn.named_parameters():
            if param.dim() > 1:  # 权重矩阵
                # if "k_b_proj" in name.lower() or "v_b_proj" in name.lower():
                #     torch.nn.init.normal_(param, mean=0.0, std=1/4)
                if 'proj' in name.lower() or 'linear' in name.lower():
                    # 不用init.xavier初始化 因为KV升维矩阵是per head进行的
                    std = math.sqrt(1 / param.shape[1])
                    torch.nn.init.normal_(param, mean=0.0, std=std)
                elif 'gate' in name.lower():
                    torch.nn.init.normal_(param, mean=0.0, std=1/4)
                else:
                    # 其他权重矩阵使用标准正态分布
                    torch.nn.init.normal_(param, mean=0.0, std=1)
            else:  # 偏置或1维参数
                if 'bias' in name.lower():
                    torch.nn.init.zeros_(param)
                else:
                    torch.nn.init.normal_(param, mean=0.0, std=1/2)

    def copy_weights(self, hf_module):
        hf_named_parameters = dict(hf_module.attn.named_parameters())
        for name, param in self.attn.named_parameters():
            assert name in hf_named_parameters
            hf_param = hf_named_parameters[name]
            default_weight_loader(param, hf_param)

class SRTModule(nn.Module):
    class DecoderCommManangerMock:
        def pre_attn_comm(self, hidden_states, tp_num_tokens, is_second_attn=False):
            return hidden_states

        def post_attn_comm(self, hidden_states, residual, tp_num_tokens, is_first_attn=True):
            return hidden_states, residual


    def __init__(self, config: AttnConfig):
        super().__init__()
        self.comm_manager = SRTModule.DecoderCommManangerMock()
        from sglang.srt.models.deepseek_mha_nsa import DeepseekNSAWithMLA as SRTDeepseekV3NSAWithMLA
        self.attn = SRTDeepseekV3NSAWithMLA(
            config = config,
            hidden_size = config.hidden_size,
            num_heads = config.num_attention_heads,
            qk_nope_head_dim = config.qk_nope_head_dim,
            qk_rope_head_dim = config.qk_rope_head_dim,
            v_head_dim = config.v_head_dim,
            q_lora_rank = config.q_lora_rank,
            kv_lora_rank = config.kv_lora_rank,
            rope_theta = config.rope_theta,
            rope_scaling = config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            quant_config = None,
            layer_id=0,
            prefix= "",
            reduce_attn_results=False,
        )
        # TODO:
        # self.attn.attention_backend = "flashinfer_mla"

    def forward(            
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        o = self.attn(
            positions=forward_batch.positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            comm_manager=self.comm_manager
        )
        return o
    
    def copy_weights(self, hf_module: HFModule):
        hf_named_parameters = dict(hf_module.attn.named_parameters())
        name_mapping = {
            "compress_kv": "compress_key",     # kv_lora_rank
            "compress_k_pe": "compress_value", # q_pe
        }
        for name, param in self.attn.named_parameters():
            for src_name, tar_name in name_mapping.items():
                name = name.replace(src_name, tar_name)
            if name in hf_named_parameters:
                hf_param = hf_named_parameters[name]
                default_weight_loader(param, hf_param)
            # elif name.startswith("kv_b_proj"):
            #     k_b_proj = hf_named_parameters["k_b_proj.weight"]
            #     v_b_proj = hf_named_parameters["v_b_proj.weight"]
            #     kv_b_proj = torch.cat([k_b_proj, v_b_proj], dim=0)
            #     default_weight_loader(param, kv_b_proj)
            elif "q_proj" in name and name not in hf_named_parameters:
                name = name.replace("q_proj", "q_a_proj")
                default_weight_loader(param, hf_named_parameters[name])
            elif name.startswith("attn.compress_attn"):
                default_weight_loader(param, hf_named_parameters[name[5:]])
            elif name.startswith("attn.gate_fusion"):
                gate_fusion = hf_named_parameters["gate_fusion.gate_weight"]
                default_weight_loader(param, gate_fusion)
            else:
                assert False, f"Unknown param {name}"

        w = self.attn.kv_b_proj.weight
        w_kc, w_vc = w.unflatten(
            0, (-1, self.attn.qk_nope_head_dim + self.attn.v_head_dim)
        ).split([self.attn.qk_nope_head_dim, self.attn.v_head_dim], dim=1)
        # [h, 128, 512]
        self.attn.w_kc = w_kc.contiguous()
        # [h, 512, 128]
        self.attn.w_vc = w_vc.contiguous().transpose(1, 2)
        self.attn.attn.w_kc = self.attn.w_kc
        self.attn.attn.w_vc = self.attn.w_vc
        # k_b, v_b 转换的方式
        # w_kc = self.attn.k_b_proj.weight
        # w_kc = w_kc.unflatten(
        #     0, (-1, self.attn.qk_nope_head_dim)
        # )
        # self.attn.attn.w_kc = w_kc.contiguous()
        # self.attn.w_kc = w_kc.contiguous()
        # w_vc = w_vc.transpose(1, 2).contiguous()
        # self.attn.attn.w_vc = w_vc
        # self.attn.w_vc = w_vc

@dataclass
class ModuleRunnerMock:
    model_config: AttnConfig
    device: torch.device
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: MLATokenToKVPool
    kv_cache_dtype: torch.dtype
    dtype: torch.dtype

def prepare_inputs(
        config: AttnConfig, 
        dtype: torch.dtype,
        seed: int, 
        seq_lens: List[int], 
        extend_prefix_lens: List[int],
    ):
    set_all_seeds(seed)

    batch_size = len(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    seq_lens_sum = seq_lens.sum().item()
    extend_prefix_lens = torch.tensor(extend_prefix_lens, dtype=torch.int32)
    extend_prefix_lens_cpu = extend_prefix_lens.tolist()
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
        max_context_len=config.max_position_embeddings + config.page_size,
        device=torch.get_default_device(),
        enable_memory_saver=False,
    )
    req_pool_indices = torch.empty((batch_size,), dtype=torch.int32)
    out_cache_loc = torch.empty((total_token_num,), dtype=torch.int32)

    allocated_tokens = 0
    allocated_new_tokens = 0
    for idx, seq_len in enumerate(seq_lens):
       new_seq_len = extend_seq_lens[idx]
       req_pool_indices[idx] = idx
       req_to_token_pool.req_to_token[idx][:seq_len] = torch.tensor(range(seq_len), dtype=torch.int32) + allocated_tokens
       out_cache_loc[allocated_new_tokens: allocated_new_tokens + new_seq_len] = req_to_token_pool.req_to_token[idx][seq_len-new_seq_len:seq_len]
       # align to next page
       allocated_tokens += seq_len + config.page_size - seq_len % config.page_size
       allocated_new_tokens += new_seq_len

    token_to_kv_pool = NativeSparseMLATokenToKVPool(
        size=seq_lens_sum + batch_size * config.page_size,
        dtype=dtype,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
        layer_num=1,
        device=torch.get_default_device(),
        enable_memory_saver=False,
        max_batch_size=batch_size,
        max_context_len=config.max_position_embeddings + config.page_size,
        page_size=config.page_size,
        rank=0,
        compressed_block_stride=config.stride
    )
   
    forward_batch = ForwardBatch(
        batch_size=batch_size,
        input_ids=None,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_sum=seq_lens_sum,
        forward_mode=forward_mode,
        positions=positions,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        out_cache_loc=out_cache_loc,
        extend_num_tokens = extend_num_tokens,
        extend_seq_lens = extend_seq_lens,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_prefix_lens = extend_prefix_lens,
        extend_prefix_lens_cpu = extend_prefix_lens_cpu,
    )
    forward_batch.tp_num_tokens = total_token_num

    hidden_states = [
        torch.randn((seq_len, config.hidden_size), dtype=dtype, device=torch.get_default_device())
        for seq_len in seq_lens
    ]
    return hidden_states, forward_batch

def fill_latent_cache(
    latent_cache: List[torch.Tensor],
    compressed_latent_cache: List[torch.Tensor],
    forward_batch: ForwardBatch,
    attn_config: AttnConfig,
):
    for seq_idx in range(len(latent_cache)):
        kv_len = forward_batch.seq_lens[seq_idx]
        req_pool_idx = forward_batch.req_pool_indices[seq_idx]
        token_indices = forward_batch.req_to_token_pool.req_to_token[req_pool_idx][:kv_len]
        forward_batch.token_to_kv_pool.kv_buffer[0][token_indices] = latent_cache[seq_idx].unsqueeze(1)
        compressed_token_indices = token_indices[:1-attn_config.kernel_size:attn_config.stride] // attn_config.stride
        forward_batch.token_to_kv_pool.compressed_kv_buffer[0][compressed_token_indices] = compressed_latent_cache[seq_idx].unsqueeze(1)

def extract_latent_cache(
    forward_batch: ForwardBatch,
    attn_config: AttnConfig,
):
    latent_cache = []
    compressed_latent_cache = []
    for seq_idx in range(len(forward_batch.seq_lens)):
        kv_len = forward_batch.seq_lens[seq_idx]
        req_pool_idx = forward_batch.req_pool_indices[seq_idx]
        token_indices = forward_batch.req_to_token_pool.req_to_token[req_pool_idx][:kv_len]
        latent_cache.append(forward_batch.token_to_kv_pool.kv_buffer[0][token_indices].squeeze(1))
        compressed_token_indices = token_indices[:1-attn_config.kernel_size:attn_config.stride] // attn_config.stride
        compressed_latent_cache.append(forward_batch.token_to_kv_pool.compressed_kv_buffer[0][compressed_token_indices].squeeze(1))

    return latent_cache, compressed_latent_cache

def prepare_for_srt(
    data: List[torch.Tensor],
    forward_batch: ForwardBatch,
):
    prepared_data = []
    for seq_idx in range(len(data)):
        if forward_batch.forward_mode.is_decode():
            new_token_num = 1
        else:
            new_token_num = forward_batch.extend_seq_lens[seq_idx].item()
        prepared_data.append(data[seq_idx][-new_token_num:])
        
    return torch.cat(prepared_data)

class TestNsaWithMla(unittest.TestCase):

    def _compare_hf_vs_srt_forward(
            self, 
            hf32_output: torch.Tensor,
            hf_output: torch.Tensor,
            srt_output: torch.Tensor,
            tolerance_factor: float = 1.5,
        ):

        assert hf32_output.shape == hf_output.shape == srt_output.shape, \
            f"形状不匹配: hf32={hf32_output.shape}, hf={hf_output.shape}, srt={srt_output.shape}"
        
        hf_output = hf_output.to(device=hf32_output.device, dtype=hf32_output.dtype)
        srt_output = srt_output.to(device=hf32_output.device, dtype=hf32_output.dtype)
        print(f"{srt_output.shape=} {hf32_output.shape=}")
        
        calc_cos = lambda a, b : 1 - 2 * (a * b).sum().item() / max((a * a + b * b).sum().item(), 1e-12)
        srt_and_32_cos_err = calc_cos(srt_output, hf32_output)
        hf_and_32_cos_err = calc_cos(hf_output, hf32_output)
        print(f"{srt_and_32_cos_err=}")
        print(f"{hf_and_32_cos_err=}")
        
        srt_diff = (srt_output - hf32_output).abs()
        hf_diff = (hf_output - hf32_output).abs()

        max_srt_error = srt_diff.max().item()
        max_hf_error = hf_diff.max().item()
        
        if max_srt_error > tolerance_factor * max_hf_error:
            max_error_idx = torch.argmax(srt_diff)
            max_error_idx_nd = torch.unravel_index(max_error_idx, hf32_output.shape)
            
            print(f"\n❌ 最大误差位置 {max_error_idx_nd}:")
            print(f"  HF32参考值: {hf32_output.flatten()[max_error_idx].item():.6e}")
            print(f"  HF输出值: {hf_output.flatten()[max_error_idx].item():.6e}")
            print(f"  SRT输出值: {srt_output.flatten()[max_error_idx].item():.6e}")
            print(f"  HF误差: {hf_diff.flatten()[max_error_idx].item():.6e}")
            print(f"  SRT误差: {srt_diff.flatten()[max_error_idx].item():.6e}")
            
            top_errors, top_indices = torch.topk(srt_diff.flatten(), k=min(5, srt_diff.numel()))
            print(f"\n前5个最大SRT误差位置:")
            for i, (error_val, error_idx) in enumerate(zip(top_errors, top_indices)):
                error_idx_nd = torch.unravel_index(error_idx, hf32_output.shape)
                print(f"  #{i+1} 位置{[c.item() for c in error_idx_nd]}: SRT值={error_val.item():.6f}, "
                    f"HF值={hf_diff.flatten()[error_idx].item():.6f}")
                
            assert False, f"SRT max error ({max_srt_error:.6e}) exceeds {tolerance_factor}x HF max error ({max_hf_error:.6e})"
    
    def load_tensor_check(self):
        hf_pt = torch.load("/tmp/hf.pt")
        fl_pt = torch.load("/tmp/fl.pt")
        calc_cos = lambda a, b : 1 - 2 * (a * b).sum().item() / max((a * a + b * b).sum().item(), 1e-12)
        for e in hf_pt:
            print(f"hf {e.shape=}")
        for e in fl_pt:
            print(f"fl {e.shape=}")
        for i in range(len(hf_pt)):
            a = hf_pt[i]
            b = fl_pt[i]
            cos_err = calc_cos(a, b)
            print(f"{cos_err=}")


    def _test_hf_vs_srt_forward(
            self, 
            attn_config: AttnConfig, 
            seed: int, 
            seq_lens: List[int], 
            extend_prefix_lens: List[int],
        ):
        assert len(seq_lens) == len(extend_prefix_lens)
        assert all(seq_len > 0 for seq_len in seq_lens)
        assert all(extend_prefix_len >= 0 for extend_prefix_len in extend_prefix_lens)
        assert all(seq_len > extend_prefix_len for seq_len, extend_prefix_len in zip(seq_lens, extend_prefix_lens))

        torch.set_default_device("cuda")
        with set_default_torch_dtype(torch.float32):
            hf32_module = HFModule(attn_config)
        with set_default_torch_dtype(torch.bfloat16):
            hf_module = HFModule(attn_config)
            srt_module = SRTModule(attn_config)
        hf_module.rand_weights(attn_config, seed)
        hf32_module.copy_weights(hf_module)
        srt_module.copy_weights(hf_module)

        with torch.no_grad():
            hidden_states, forward_batch = prepare_inputs(attn_config, torch.bfloat16, seed, seq_lens, extend_prefix_lens)
            hidden_states_32 = [hidden_state.to(torch.float32) for hidden_state in hidden_states]
        
            hf_output, hf_latent_cache, hf_compressed_latent_cache = hf_module(hidden_states)
            hf32_output, hf32_latent_cache, hf32_compressed_latent_cache = hf32_module(hidden_states_32)

            fill_latent_cache(hf_latent_cache, hf_compressed_latent_cache, forward_batch, attn_config)
            hidden_states = prepare_for_srt(hidden_states, forward_batch)
            hf32_output = prepare_for_srt(hf32_output, forward_batch)
            hf_output = prepare_for_srt(hf_output, forward_batch)
            srt_output = srt_module(hidden_states, forward_batch)
            srt_latent_cache, srt_compressed_latent_cache = extract_latent_cache(forward_batch, attn_config)

            # self.load_tensor_check()

            self._compare_hf_vs_srt_forward(
                hf32_output = hf32_output,
                hf_output = hf_output,
                srt_output = srt_output,
            )
            self._compare_hf_vs_srt_forward(
                hf32_output = torch.cat(hf32_latent_cache),
                hf_output = torch.cat(hf_latent_cache),
                srt_output = torch.cat(srt_latent_cache),
            )
            self._compare_hf_vs_srt_forward(
                hf32_output = torch.cat(hf32_compressed_latent_cache),
                hf_output = torch.cat(hf_compressed_latent_cache),
                srt_output = torch.cat(srt_compressed_latent_cache),
            )
    @unittest.skip
    def test_prefill(self):
        seed = 42
        attn_config = AttnConfig()
        max_seq_len = 8192
        for batch_size in [1, 2, 16, 32]:
            set_all_seeds(seed)
            seq_lens = [ random.randint(1, max_seq_len) for _ in range(batch_size) ]
            extend_prefix_lens = [0] * batch_size
            with self.subTest(seed=seed, batch_size=batch_size):
                self._test_hf_vs_srt_forward(attn_config, seed, seq_lens, extend_prefix_lens)
    # @unittest.skip
    def test_decode(self):
        seed = 42
        attn_config = AttnConfig()
        max_seq_len = 8192
        for batch_size in [1, 2, 16, 32]:
            set_all_seeds(seed)
            seq_lens = [ random.randint(1, max_seq_len) for _ in range(batch_size) ]
            extend_prefix_lens = [ seq_len -1 for seq_len in seq_lens]
            with self.subTest(seed=seed, batch_size=batch_size):
                self._test_hf_vs_srt_forward(attn_config, seed, seq_lens, extend_prefix_lens)
    @unittest.skip
    def test_extend(self):
        seed = 42
        attn_config = AttnConfig()
        max_seq_len = 8192
        for batch_size in [1, 2, 16]:
            set_all_seeds(seed)
            seq_lens = [ random.randint(3, max_seq_len) for _ in range(batch_size) ]
            seq_lens = [5241]
            extend_prefix_lens = [ seq_len // 2 for seq_len in seq_lens]
            with self.subTest(seed=seed, batch_size=batch_size):
                self._test_hf_vs_srt_forward(attn_config, seed, seq_lens, extend_prefix_lens)


if __name__ == '__main__':
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
        global_rank=None,
        local_rank=0,
        hidden_size=1,
        max_num_tokens=1,
    )
    try:
        unittest.main()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()

