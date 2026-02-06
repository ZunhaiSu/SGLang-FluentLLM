import math
import random
import unittest
import torch
import torch.profiler
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.linear import ReplicatedLinear

from sglang.srt.layers.attention.native_sparse_attention.compress_kv import (
    gate_compress_decode_torch,
    gate_compress_decode_triton,
)

class TestCompressKV(unittest.TestCase):
    # Test constants
    DEFAULT_SEED = 42
    DEFAULT_TOLERANCE_RTOL = 0.02
    DEFAULT_TOLERANCE_ATOL = 1e-3
    
    def _set_all_seeds(self, seed):
        """Set all random seeds to ensure reproducible results"""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setUp(self):
        """Test environment initialization"""
        self._set_all_seeds(self.DEFAULT_SEED)
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.float32)

    def _create_test_data(self, seq_lens, kv_head_num, head_dim, kernel_size):
        """Create test data
        
        Args:
            seq_lens: List of sequence lengths, e.g. [128, 256]
            kv_head_num: Number of KV attention heads
            head_dim: Dimension of each head
            kernel_size: Compression kernel size
            
        Returns:
            Dictionary containing all test data
        """
        total_kv_len = sum(seq_lens)
        max_seq_len = max(seq_lens) if seq_lens else 0
        batch_size = len(seq_lens)
        
        # Create KV cache (randomly initialized to simulate real data)
        key_cache = torch.randn(total_kv_len, kv_head_num, head_dim)
        
        # Create sequence lengths and index mappings
        kv_lens = torch.tensor(seq_lens, dtype=torch.int32)
        req_pool_indices = torch.arange(batch_size, dtype=torch.int32)
        
        # Create token index mapping table
        req_to_token = torch.zeros(batch_size, max_seq_len, dtype=torch.int32)
        cumsum_lens = 0
        for i, seq_len in enumerate(seq_lens):
            req_to_token[i, :seq_len] = torch.arange(cumsum_lens, cumsum_lens + seq_len)
            cumsum_lens += seq_len
            
        # Create gate projection layer (for generating compression weights)
        gate_proj = ReplicatedLinear(
            input_size=kernel_size * head_dim,
            output_size=kernel_size,
            bias=False,
        )
        torch.nn.init.normal_(gate_proj.weight, std=(kernel_size * head_dim)**-0.5)
        
        return {
            'key_cache': key_cache,
            'kv_lens': kv_lens,
            'req_pool_indices': req_pool_indices,
            'req_to_token': req_to_token,
            'gate_proj': gate_proj,
        }

    def _test_compress_kv_decode_common(self, seq_lens, kv_head_num, head_dim, kernel_size, kernel_stride):
        """Common compress KV test function
        
        Args:
            seq_lens: List of sequence lengths
            kv_head_num: Number of KV heads  
            head_dim: Head dimension
            kernel_size: Compression kernel size
            kernel_stride: Compression stride
        """

        test_data = self._create_test_data(seq_lens, kv_head_num, head_dim, kernel_size)

        compressed_key_cache = torch.ones(math.ceil(sum(seq_lens) / kernel_stride), kv_head_num, head_dim)
        ref_compressed_key_cache = torch.ones_like(compressed_key_cache)
        
        try:
            # Run Triton implementation
            gate_compress_decode_triton(
                kv_cache=test_data['key_cache'],
                compressed_kv_cache=compressed_key_cache,
                gate_weight=test_data['gate_proj'].weight,
                kv_lens=test_data['kv_lens'],
                req_pool_indices=test_data['req_pool_indices'],
                req_to_token=test_data['req_to_token'],
                kernel_size=kernel_size,
                stride=kernel_stride,
            )
            
            # Run PyTorch reference implementation
            gate_compress_decode_torch(
                kv_buffer=test_data['key_cache'],
                compressed_kv_buffer=ref_compressed_key_cache,
                gate_proj=test_data['gate_proj'],
                kv_lens=test_data['kv_lens'],
                req_pool_indices=test_data['req_pool_indices'],
                req_to_token=test_data['req_to_token'],
                kernel_size=kernel_size,
                stride=kernel_stride,
            )
            
            # Verify result consistency
            is_close = torch.allclose(
                compressed_key_cache, 
                ref_compressed_key_cache, 
                rtol=self.DEFAULT_TOLERANCE_RTOL, 
                atol=self.DEFAULT_TOLERANCE_ATOL
            )
            
                
            self.assertTrue(is_close)
            
        except Exception as e:
            print(f"  Exception: {str(e)}")
            raise

    def test_compress_kv_decode_seq_lens_variations(self):
        """Test group for sequence length variations"""
        test_cases = [
            # (test_name, seq_lens, kv_head_num, head_dim, kernel_size, kernel_stride)
            ("single_sequence", [64], 16, 128, 32, 16),
            ("multiple_sequences", [48, 64, 80, 96], 16, 128, 32, 16),
            ("unaligned_sequences", [16-1, 32 -1, 32+16-1], 16, 128, 32, 16),
            ("long_sequence", [32 + 16*128 - 1, 32 + 16*128], 16, 128, 32, 16),
        ]
        
        for test_name, seq_lens, kv_head_num, head_dim, kernel_size, kernel_stride in test_cases:
            with self.subTest(test_case=test_name):
                self._test_compress_kv_decode_common(
                    seq_lens=seq_lens,
                    kv_head_num=kv_head_num,
                    head_dim=head_dim,
                    kernel_size=kernel_size,
                    kernel_stride=kernel_stride,
                )

    def test_compress_kv_decode_head_variations(self):
        """Test group for head count and dimension variations"""
        test_cases = [
            # (test_name, seq_lens, kv_head_num, head_dim, kernel_size, kernel_stride)
            ("minimal_heads", [32], 1, 128, 32, 16),
            ("many_heads", [32], 128, 128, 32, 16),
            ("small_head_dim", [32], 16, 8, 32, 16),
            ("large_head_dim", [32], 16, 512, 32, 16),
        ]
        
        for test_name, seq_lens, kv_head_num, head_dim, kernel_size, kernel_stride in test_cases:
            with self.subTest(test_case=test_name):
                self._test_compress_kv_decode_common(
                    seq_lens=seq_lens,
                    kv_head_num=kv_head_num,
                    head_dim=head_dim,
                    kernel_size=kernel_size,
                    kernel_stride=kernel_stride,
                )

    def test_compress_kv_decode_kernel_variations(self):
        """Test group for kernel size and stride variations"""
        test_cases = [
            # (test_name, seq_lens, kv_head_num, head_dim, kernel_size, kernel_stride)
            ("non_power_of_two", [32 + 2*11, 32 + 2*11 -1], 16, 128, 32, 11),
            ("stride_equals_kernel", [2*32, 2* 32 -1], 16, 128, 32, 32),
            ("stride_larger_than_kernel", [32+64, 32+64-1], 16, 128, 32, 64),
            ("small_kernel_and_stride", [32, 32+4], 16, 128, 16, 4),
            ("large_kernel_and_stride", [1024 + 2*256], 16, 128, 128, 64),
        ]
        
        for test_name, seq_lens, kv_head_num, head_dim, kernel_size, kernel_stride in test_cases:
            with self.subTest(test_case=test_name):
                self._test_compress_kv_decode_common(
                    seq_lens=seq_lens,
                    kv_head_num=kv_head_num,
                    head_dim=head_dim,
                    kernel_size=kernel_size,
                    kernel_stride=kernel_stride,
                )

if __name__ == '__main__':
    unittest.main()