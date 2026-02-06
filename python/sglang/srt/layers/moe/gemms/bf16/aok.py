from typing import Optional

from eps.executor import AokGroupedGemm

from sglang.srt.layers.moe.config import DispatcherType, EPConfig
from sglang.srt.layers.activation import SwigluArg
from sglang.srt.layers.moe.executors.eps_executor import EPSExecutor
from sglang.srt.layers.moe.layouts.unquant import UnquantMoELayout


class AokGroupedGemmWrapper:
    def __init__(self, top_k, num_experts, layer, quant_config):
        self.top_k = top_k
        self.num_experts = num_experts
        num_local_experts, _, hidden_size = layer.w13_weight.shape
        self.hidden_size = hidden_size

        self.gemm = AokGroupedGemm(num_local_experts)

    @staticmethod
    def get_layout_cls():
        return UnquantMoELayout

    def get_executor(
        self,
        dispatcher_type: DispatcherType,
        activation: str,
        swiglu_arg: Optional[SwigluArg] = None,
    ):
        if dispatcher_type == DispatcherType.EPS:
            ep_config = EPConfig.from_gemm_wrapper(self)
            return EPSExecutor(self.gemm, ep_config, activation, swiglu_arg)
        else:
            raise RuntimeError("To implement")
