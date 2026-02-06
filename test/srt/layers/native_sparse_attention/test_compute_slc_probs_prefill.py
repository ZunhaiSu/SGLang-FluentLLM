import os, sys
import random
import unittest
import math
import torch
from sglang.srt.layers.attention.native_sparse_attention.select_attn import (
    compute_select_probs
)

class TestSelectProbsPrefill(unittest.TestCase):
    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(42)
    
    def test_probs_equivalence(self):
        torch.set_default_device('cuda')
        com_stride = 16
        com_size = 32
        slc_size = 64
        for i, kvlen in enumerate([4043]*8):
            num_selcct_block = math.ceil(kvlen / slc_size)
            num_com_block = (kvlen - com_size) // com_stride + 1
            com_probs = torch.ones(1,1,1,num_com_block, device='cuda') * 1
            com_probs[..., -4:] = 10
            slc_probs = torch.zeros(1,1,1,num_selcct_block, device='cuda')
            for select_idx in range(num_selcct_block):
                acc_probs = slc_probs[:, :, :, select_idx]
                select_start = select_idx * slc_size
                select_end = min(select_start + slc_size, kvlen)
                # compress_start_idx is idx in `p`, 第几个com_block
                compress_start_idx = max((select_start - com_size) // com_stride + 1, 0)
                compress_start = compress_start_idx * com_stride
                while compress_start < select_end and compress_start + com_size <= kvlen:
                    compress_end = compress_start + com_size
                    area = min(compress_end, select_end) - max(compress_start, select_start)
                    acc_probs += com_probs[:, :, :, compress_start_idx] * area / com_stride
                    compress_start_idx += 1
                    compress_start += com_stride
            q = torch.empty(1,1,1,128, device='cuda')
            k = torch.empty(1,1,kvlen,128, device='cuda')
            slc_probs_triton, _, _ = compute_select_probs(
                com_probs, com_size, com_stride, slc_size, 16, kvlen, 1
            )

            self.assertTrue(
                torch.allclose(slc_probs, slc_probs_triton, rtol=1e-1, atol=1e-2),
                msg=f"Failed at i={i}\nslc_probs={slc_probs}\nslc_probs_triton={slc_probs_triton}")

if __name__ == '__main__':
    unittest.main()
