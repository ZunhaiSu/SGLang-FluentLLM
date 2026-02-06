import torch
import triton
import triton.language as tl


@triton.jit
def rearrange_accept_index(
    accept_index_ptr,
    accept_length_ptr,
    output_ptr,
    num_tokens_per_req_upper: tl.constexpr,
    accept_index_stride: tl.constexpr,
):
    pid = tl.program_id(0)
    accept_len = tl.load(accept_length_ptr + pid)
    cum_accept_len = 0
    for i in range(pid):
        cum_accept_len += tl.load(accept_length_ptr + i)
    store_offset = tl.arange(0, num_tokens_per_req_upper)
    accept_index_load_offset = (
        tl.arange(0, num_tokens_per_req_upper) + pid * accept_index_stride
    )
    accept_index = tl.load(accept_index_ptr + accept_index_load_offset)
    tl.store(
        output_ptr + store_offset + cum_accept_len,
        accept_index,
        mask=store_offset < accept_len,
    )


if __name__ == "__main__":
    bs = 5
    draft_token_num = 4
    predict_ids = torch.tensor(
        [i for i in range(bs * draft_token_num)], dtype=torch.int32, device="cuda:0"
    )
    accept_index = predict_ids.clone().reshape(bs, draft_token_num)
    accept_length = torch.tensor([4, 1, 4, 3, 2], dtype=torch.int32, device="cuda:0")
    accept_index[1, 1:] = -1
    accept_index[4, 2:] = -1
    accept_index[3, -1] = -1

    accept_index_rearranged = torch.zeros_like(predict_ids)
    rearrange_accept_index[(bs,)](
        accept_index,
        accept_length,
        accept_index_rearranged,
        triton.next_power_of_2(draft_token_num),
        accept_index.shape[1],
    )
    print(accept_index_rearranged)
    print(accept_index)
    assert torch.equal(accept_index_rearranged, accept_index[accept_index != -1])