import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import get_tp_group, get_cross_group_from_list, get_local_group_from_list
from sglang.srt.utils import is_npu


two_stage_comm = 1

def reduce_scatter_two_stage(input_: torch.Tensor, idx: int, reverse=False) -> torch.Tensor:
    if two_stage_comm == 0:
        # TODO: multi comm
        # return get_world_group_from_list(idx).reduce_scatter(input_)
        return get_tp_group().reduce_scatter(input_)
    if reverse:
        stage1 = get_cross_group_from_list(idx).reduce_scatter(input_)
        return get_local_group_from_list(idx).reduce_scatter(stage1)
    stage1 = get_local_group_from_list(idx).reduce_scatter(input_)
    return get_cross_group_from_list(idx).reduce_scatter(stage1)


def all_gather_two_stage(input_: torch.Tensor, idx: int, dim=-1, reverse=False) -> torch.Tensor:
    if two_stage_comm == 0:
        # TODO: multi comm
        # return get_world_group_from_list(idx).all_gather(input_, dim)
        return get_tp_group().all_gather(input_, dim)
    if reverse:
        stage1 = get_local_group_from_list(idx).all_gather(input_, dim)
        return get_cross_group_from_list(idx).all_gather(stage1, dim)
    stage1 = get_cross_group_from_list(idx).all_gather(input_, dim)
    return get_local_group_from_list(idx).all_gather(stage1, dim)


def reduce_scatter_local(input_: torch.Tensor, idx: int) -> torch.Tensor:
    return get_local_group_from_list(idx).reduce_scatter(input_)


def reduce_scatter_cross(input_: torch.Tensor, idx: int) -> torch.Tensor:
    return get_cross_group_from_list(idx).reduce_scatter(input_)


def all_gather_local(input_: torch.Tensor, idx: int, dim=-1) -> torch.Tensor:
    return get_local_group_from_list(idx).all_gather(input_, dim)

def all_gather_cross(input_: torch.Tensor, idx: int, dim=-1) -> torch.Tensor:
    return get_cross_group_from_list(idx).all_gather(input_, dim)

class NpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not is_npu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if dim < 0:
            # Convert negative dim to positive.
            dim += x.dim()
        input_size = x.size()
        output_size = (input_size[0] * world_size,) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size, dtype=x.dtype, device=x.device)
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, x, group=self.group)
        # Reshape
        output_tensor = output_tensor.reshape((world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        dist.all_gather_into_tensor(output, input, group=self.group)
