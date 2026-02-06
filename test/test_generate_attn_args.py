import torch

from sglang.srt.speculative.eagle_utils import generate_attn_arg_prefill, generate_attn_arg_v2

def test_compare_attn_args():
    batch_size = 4
    max_seq_len = 128
    draft_token_num = 4
    
    paged_kernel_lens = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device="cuda")
    req_to_token = torch.arange(batch_size * max_seq_len, dtype=torch.int32, device="cuda").view(batch_size, max_seq_len)
    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    kv_indices_buf_v2 = torch.full((2048,), -1, dtype=torch.int32, device="cuda")
    
    kv_indices_p, cum_kv_p, qo_p, _ = generate_attn_arg_prefill(
        draft_token_num, req_pool_indices, paged_kernel_lens.clone(), req_to_token, None, None
    )
    
    kv_indices_v2, cum_kv_v2, qo_v2 = generate_attn_arg_v2(
        draft_token_num, req_pool_indices, paged_kernel_lens.clone(), req_to_token, kv_indices_buf_v2.clone(), False, None
    )

    print(f"QO Indptr Equal: {torch.equal(qo_p, qo_v2)}")
    print(f"Cum KV Lens Equal: {torch.equal(cum_kv_p, cum_kv_v2)}")
    
    valid_len_p = cum_kv_p[-1]
    valid_len_v2 = cum_kv_v2[-1]
    
    if valid_len_p == valid_len_v2:
        match = torch.equal(kv_indices_p[:valid_len_p], kv_indices_v2[:valid_len_v2])
        print(f"KV Indices Content Match: {match}")
        if not match:
            print("Mismatch Detail (Last 8 elements):")
            print(f"Prefill: {kv_indices_p[valid_len_p-8:valid_len_p]}")
            print(f"V2:      {kv_indices_v2[valid_len_v2-8:valid_len_v2]}")
    else:
        print(f"Length Mismatch: Prefill={valid_len_p}, V2={valid_len_v2}")

    for i in range(4):
        kv_indices_p, cum_kv_p, qo_p, _ = generate_attn_arg_prefill(
            draft_token_num, req_pool_indices, paged_kernel_lens.clone(), req_to_token, None, None
        )
        
        kv_indices_v2, cum_kv_v2, qo_v2 = generate_attn_arg_v2(
            draft_token_num, req_pool_indices, paged_kernel_lens.clone(), req_to_token, kv_indices_buf_v2.clone(), True, i 
        )
        valid_len_p = cum_kv_p[-1]
        valid_len_v2 = cum_kv_v2[-1]
        
        if valid_len_p == valid_len_v2:
            match = torch.equal(kv_indices_p[:valid_len_p], kv_indices_v2[:valid_len_v2])
            print(f"KV Indices Content Match: {match}")
            if not match:
                print("Mismatch Detail (Last 8 elements):")
                print(f"Prefill: {kv_indices_p[valid_len_p-8:valid_len_p]}")
                print(f"V2:      {kv_indices_v2[valid_len_v2-8:valid_len_v2]}")
        else:
            print(f"Length Mismatch: Prefill={valid_len_p}, V2={valid_len_v2}")

if __name__ == "__main__":
    test_compare_attn_args()