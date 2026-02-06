import os, sys
import math
import random
import unittest
import torch
import torch.profiler
from sglang.srt.models.qwen3_nsa import (
    NativeSparseAttention
)
from sglang.srt.layers.attention.native_sparse_attention.select_attn import (
    select_attention_decode_triton, _select_attention_torch_aligned,
    select_attention_decode_splitk_triton
)
from sglang.srt.layers.attention.native_sparse_attention.compress_attn import (
    _transform_score_decode_triton, _fill_topkidx_decode_triton
)

class TestSelectAttn(unittest.TestCase):
    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setUp(self):
        self._set_all_seeds(42)

    def test_select_attn(self):
        torch.set_default_device("cuda")
        dtype = torch.bfloat16
        kvcache_len = 4096
        kv_head_num = 1
        head_dim = 128
        q_head_num = kv_head_num
        key_cache = torch.randn(
            kvcache_len, kv_head_num, head_dim, dtype=dtype
        )
        value_cache = torch.randn(
            kvcache_len, kv_head_num, head_dim, dtype=dtype
        )
        select_size = 64
        top_n = 16
        seq_lens = [1775]
        seq_num = len(seq_lens)
        # decode
        q = torch.randn(
            seq_num, q_head_num, head_dim, dtype=dtype
        )
        kv_locs = []
        top_indices_list = []
        topk_indices_tensor = []
        cur_len = 64
        for seq_len in seq_lens:
            kv_locs.append(torch.arange(cur_len, cur_len+seq_len, dtype=torch.int64))
            cur_len += seq_len
            num_selcct_blocks = math.ceil(seq_len / select_size)
            tmp_ind = torch.randperm(num_selcct_blocks)
            # 最多选top_n
            top_indices = torch.arange(num_selcct_blocks)[tmp_ind][:top_n]
            # 右padding
            topk_indices_tensor.append(
                torch.nn.functional.pad(top_indices, (0, top_n-top_indices.shape[-1]), mode='constant', value=-1)[None])
            top_indices = top_indices.reshape(1, 1, 1, -1)\
                .expand(1, kv_head_num, 1, -1)
            top_indices_list.append(top_indices)
        
        slc_torch_o = []
        for seq_i in range(seq_num):
            print(f"{top_indices_list[seq_i]=}")
            args = [
                q[seq_i][None, None], # [1, qlen, qh, hd]
                key_cache[kv_locs[seq_i]][None],  # [1, seq_len, kvh, hd]
                value_cache[kv_locs[seq_i]][None] # [1, seq_len, kvh, hd]
            ]
            # args = list(map(lambda e: e.to(torch.float32), args))
            slc_o = _select_attention_torch_aligned(
                *args,
                top_indices_list[seq_i], # [1, kvh, qlen, slc_block_num]
                sm_scale=head_dim ** -0.5,
                select_size=select_size,
                num_slc_score_heads=kv_head_num
            )
            # torch.save(args + [top_indices_list[seq_i], slc_o], "/home/wuguanyu02/tensors/0719.pt")
            slc_torch_o.append(slc_o[0])
        slc_torch_o = torch.cat(slc_torch_o, dim=0)
        
        kv_indptr = torch.zeros(seq_num+1, dtype=torch.int64)
        kv_indptr[1:] = torch.cumsum(torch.tensor(seq_lens), dim=0)
        kv_indices = torch.cat(kv_locs)
        # [seq_num, topk]
        topk_indices_tensor = torch.cat(topk_indices_tensor)
        print(f"{topk_indices_tensor=}")
        # [bs, kvhead, topk]
        topk_indices_tensor = topk_indices_tensor.unsqueeze(1).expand(seq_num, kv_head_num, top_n)
        # [seq_num, qh, hd]
        slc_triton_o = select_attention_decode_triton(
            q, key_cache, value_cache, topk_indices_tensor, kv_indptr, kv_indices,
            head_dim ** -0.5, select_size
        )
        slc_torch_o = slc_torch_o[0]
        slc_triton_o = slc_triton_o[0]
        # torch.set_printoptions(threshold=float('inf'))
        # torch.set_printoptions(sci_mode=False)
        # topk_err_val, topk_ind = torch.topk(torch.abs(slc_torch_o - slc_triton_o), 4)
        # print(f"{topk_err_val=}")
        # print(torch.gather(slc_torch_o, 1, topk_ind))
        # print(torch.gather(slc_triton_o, 1, topk_ind))
        self.assertTrue(slc_torch_o.shape == slc_triton_o.shape)
        self.assertTrue(torch.allclose(slc_triton_o, slc_torch_o, rtol=2e-2, atol=0.01))
    
    def test_select_attn_splitk(self):
        torch.set_default_device("cuda")
        dtype = torch.bfloat16
        kvcache_len = 16384
        kv_head_num = 1
        head_dim = 128
        q_head_num = kv_head_num
        key_cache = torch.randn(
            kvcache_len, kv_head_num, head_dim, dtype=dtype
        )
        value_cache = torch.randn(
            kvcache_len, kv_head_num, head_dim, dtype=dtype
        )
        select_size = 64
        top_n = 16
        seq_lens = [3, 78, 2177, 8199]
        seq_num = len(seq_lens)
        # decode
        q = torch.randn(
            seq_num, q_head_num, head_dim, dtype=dtype
        )
        kv_locs = []
        top_indices_list = []
        topk_indices_tensor = []
        cur_len = 64
        for seq_len in seq_lens:
            kv_locs.append(torch.arange(cur_len, cur_len+seq_len, dtype=torch.int64))
            cur_len += seq_len
            num_selcct_blocks = math.ceil(seq_len / select_size)
            tmp_ind = torch.randperm(num_selcct_blocks)
            # 最多选top_n
            top_indices = torch.arange(num_selcct_blocks)[tmp_ind][:top_n]
            # 右padding
            topk_indices_tensor.append(
                torch.nn.functional.pad(top_indices, (0, top_n-top_indices.shape[-1]), mode='constant', value=-1)[None])
            top_indices = top_indices.reshape(1, 1, 1, -1)\
                .expand(1, kv_head_num, 1, -1)
            top_indices_list.append(top_indices)
        
        slc_torch_o = []
        for seq_i in range(seq_num):
            print(f"{top_indices_list[seq_i]=}")
            args = [
                q[seq_i][None, None], # [1, qlen, qh, hd]
                key_cache[kv_locs[seq_i]][None],  # [1, seq_len, kvh, hd]
                value_cache[kv_locs[seq_i]][None] # [1, seq_len, kvh, hd]
            ]
            # args = list(map(lambda e: e.to(torch.float32), args))
            slc_o = _select_attention_torch_aligned(
                *args,
                top_indices_list[seq_i], # [1, kvh, qlen, slc_block_num]
                sm_scale=head_dim ** -0.5,
                select_size=select_size,
                num_slc_score_heads=kv_head_num
            )
            slc_torch_o.append(slc_o[0])
        slc_torch_o = torch.cat(slc_torch_o, dim=0)
        
        kv_indptr = torch.zeros(seq_num+1, dtype=torch.int64)
        kv_indptr[1:] = torch.cumsum(torch.tensor(seq_lens), dim=0)
        kv_indices = torch.cat(kv_locs)
        # [seq_num, topk]
        topk_indices_tensor = torch.cat(topk_indices_tensor)
        print(f"{topk_indices_tensor=}")
        # [bs, kvhead, topk]
        topk_indices_tensor = topk_indices_tensor.unsqueeze(1).expand(seq_num, kv_head_num, top_n)
        quick_launch_triton = lambda: select_attention_decode_triton(
            q, key_cache, value_cache, topk_indices_tensor, kv_indptr, kv_indices,
            head_dim ** -0.5, select_size
        )
        # [seq_num, qh, hd]
        slc_triton_o = quick_launch_triton()
        quick_launch_splitk_triton = lambda: select_attention_decode_splitk_triton(
            q, key_cache, value_cache, topk_indices_tensor, kv_indptr, kv_indices,
            head_dim ** -0.5, select_size, 8
        )
        slc_splitk_triton_o = quick_launch_splitk_triton()
        
        self.assertTrue(torch.allclose(slc_triton_o, slc_torch_o, rtol=2e-2, atol=0.01))
        # print(f"{slc_triton_o[0, 0]=}")
        # print(f"{slc_splitk_triton_o[0, 0]=}")
        # print(f"{slc_triton_o[1, 0]=}")
        # print(f"{slc_splitk_triton_o[1, 0]=}")
        self.assertTrue(torch.allclose(slc_triton_o, slc_splitk_triton_o, rtol=2e-2, atol=0.01))

    def get_time_cost_ms(self, func, *args, **kargs):
        # 这个始终测不太准，因为torch的空隙太大，除非开cuda graph？
        # torch_profiler = torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # )
        # torch_profiler.start()
        # torch_profiler.stop()
        # torch_profiler.export_chrome_trace("trace.json")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for i in range(4):
            func(*args, **kargs)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / 4

    @unittest.skip
    def test_transformer_score(self):
        torch.set_default_device("cuda")
        dtype = torch.float32
        kv_head_num = 2
        head_dim = 128
        com_stride = 16
        com_size = 32
        select_size = 64
        num_init_blocks = 1
        num_local_blocks = 2
        assert com_size%com_stride==0 and select_size%com_stride==0
        bs = 1
        # seq_lens = [random.randint(1794, 1796) for i in range(bs)]
        seq_lens = [1774]
        com_block_nums = [max(0, (l-com_size)//com_stride + 1) for l in seq_lens]
        slc_block_nums = [math.ceil(l / select_size) for l in seq_lens]
        print(f"{seq_lens=}")
        print(f"{com_block_nums=}")
        print(f"{slc_block_nums=}")
        flat_com_scores = []
        ref_flat_slc_scores = []
        for seq_i in range(len(seq_lens)):
            kv_len = seq_lens[seq_i]
            slc_block_num = slc_block_nums[seq_i]
            com_block_num = com_block_nums[seq_i]
            com_score = torch.randn(com_block_num, kv_head_num, dtype=dtype)
            slc_score = torch.zeros(slc_block_num, kv_head_num, dtype=dtype)
            for slc_i in range(slc_block_num):
                acc_probs = slc_score[slc_i, :]
                select_start = slc_i * select_size
                select_end = min(select_start + select_size, kv_len)
                compress_start_idx = max((select_start - com_size) // com_stride + 1, 0)
                compress_start = compress_start_idx * com_stride
                while compress_start < select_end and compress_start + com_size <= kv_len:
                    compress_end = compress_start + com_size
                    area = min(compress_end, select_end) - max(compress_start, select_start)
                    acc_probs += com_score[compress_start_idx] * area / com_stride
                    compress_start_idx += 1
                    compress_start += com_stride
                if slc_i < num_init_blocks:
                    acc_probs += 9999 - acc_probs
                if slc_i >= slc_block_num - num_local_blocks:
                    acc_probs += 9999 - acc_probs
            # print(f"{com_score.flatten()=}")
            # print(f"{slc_score.flatten()=}")
            flat_com_scores.append(com_score)
            ref_flat_slc_scores.append(slc_score)
        flat_com_scores = torch.cat(flat_com_scores, dim=0)
        ref_flat_slc_scores = torch.cat(ref_flat_slc_scores, dim=0)
        flat_slc_scores = torch.empty(sum(slc_block_nums), kv_head_num, dtype=torch.float32)
        _transform_score_decode_triton(
            flat_com_scores, kv_head_num,
            torch.tensor(com_block_nums, dtype=torch.int32),
            torch.tensor(slc_block_nums, dtype=torch.int32),
            com_stride, com_size, select_size,
            num_init_blocks, num_local_blocks,
            flat_slc_scores
        )
        # print(f"{ref_flat_slc_scores[:slc_block_nums[0], :6]=}")
        # print(f"{flat_slc_scores[:, :6]=}")
        self.assertTrue(torch.allclose(ref_flat_slc_scores, flat_slc_scores, rtol=2e-2))
    @unittest.skip
    def test_fill_topk(self):
        torch.set_default_device("cuda")
        dtype = torch.float32
        kv_head_num = 5
        topk = 16
        select_size = 64
        flat_slc_score = torch.randn(28, kv_head_num, dtype=dtype)
        flat_slc_score[:1] = 9999
        flat_slc_score[-2:] = 9999
        topk_idx = torch.empty(1, kv_head_num, topk, dtype=torch.int32)
        _fill_topkidx_decode_triton(
            flat_slc_score,
            torch.tensor([28], dtype=torch.int32),
            topk_idx,
            max_slc_blocks_per_seq=512,
        )
        topk_idx, _ = topk_idx.sort()
        _, ref_topk_idx = flat_slc_score.topk(topk, 0)
        ref_topk_idx, _ = ref_topk_idx.transpose(0, 1).sort()
        ref_topk_idx = ref_topk_idx[None,].to(torch.int32)
        self.assertTrue(topk_idx.shape == ref_topk_idx.shape)
        self.assertTrue(torch.allclose(topk_idx, ref_topk_idx))

if __name__ == '__main__':
    unittest.main()
