

from typing import Callable

from sglang.srt.layers.dense.gemms.fp8.cutlass import CUTLASS_BLOCK_FP8_SUPPORTED, cutlass_w8a8_block_fp8_linear_with_fallback
from sglang.srt.layers.dense.gemms.fp8.deep_geem import deepgemm_w8a8_block_fp8_linear_with_fallback
from sglang.srt.layers.dense.gemms.fp8.flashinfer import ENABLE_FLASHINFER_GEMM, flashinfer_gemm_w8a8_block_fp8_linear
from sglang.srt.layers.dense.gemms.fp8.triton import triton_w8a8_block_fp8_linear


def dispatch_w8a8_block_fp8_linear() -> Callable:
    if ENABLE_FLASHINFER_GEMM:
        return flashinfer_gemm_w8a8_block_fp8_linear
    elif CUTLASS_BLOCK_FP8_SUPPORTED:
        return cutlass_w8a8_block_fp8_linear_with_fallback
    else:
        return deepgemm_w8a8_block_fp8_linear_with_fallback
