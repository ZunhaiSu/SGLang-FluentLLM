from typing import Optional

from deep_gemm import m_grouped_gemm_fp8_fp8_bf16_nt_offset

from sglang.srt.layers.moe.config import DispatcherType, EPConfig
from sglang.srt.layers.activation import SwigluArg
from sglang.srt.layers.moe.executors.fp8_eps_executor import FP8EPSExecutor
from sglang.srt.layers.moe.layouts.fp8 import Fp8MoEBlockQuantLayout


class FireGroupedGemmWrapper:
    def __init__(self, top_k, num_experts, layer, quant_config):
        self.top_k= top_k
        self.num_experts= num_experts
        num_local_experts, _, hidden_size = layer.w13_weight.shape
        self.hidden_size = hidden_size

        self.gemm = m_grouped_gemm_fp8_fp8_bf16_nt_offset

    @staticmethod
    def get_layout_cls():
        return Fp8MoEBlockQuantLayout

    def get_executor(self, dispatcher_type : DispatcherType, activation: str, swiglu_arg: Optional[SwigluArg] = None):
        if dispatcher_type == DispatcherType.EPS:
            ep_config = EPConfig.from_gemm_wrapper(self)
            return FP8EPSExecutor(self.gemm, ep_config, activation, swiglu_arg)
        else:
            raise RuntimeError("To implement")
