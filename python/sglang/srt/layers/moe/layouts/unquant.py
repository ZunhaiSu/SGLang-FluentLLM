from torch.nn import Module
from sglang.srt.layers.moe.layouts.common import MoELayout


class UnquantMoELayout(MoELayout):
    def __init__(self, quant_config):
        assert quant_config is None, f"{quant_config=}"
        super().__init__()

    def process_weights_after_loading(self, layer: Module) -> None:
        return
