# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode
from sglang.srt.model_executor.model_runner import LogitsProcessorOutput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_available_gpu_memory, get_colorful_logger
from sglang.srt.oe_utils import update_token_table
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    generate_token_bitmask,
)
from flashinfer import ngram_matching

logger = get_colorful_logger(__name__)


class PLDWorker:
    """
    Prompt Lookup Decode worker that uses n-gram matching for speculative decoding.
    Unlike EAGLE, this doesn't require a separate draft model.
    """

    def __init__(
        self,
        server_args,
        gpu_id: int,
        attn_tp_rank: int,
        dp_rank: int,
        nccl_port: int,
        target_worker,
        global_rank: int,
    ):
        self.server_args = server_args
        self.target_worker = target_worker
        self.device = target_worker.device
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.use_over_embedding = self.target_worker.model_runner.use_over_embedding

        logger.info(
            f"PLDWorker initialized with prompt_lookup_min={server_args.prompt_lookup_min}, "
            f"prompt_lookup_max={server_args.prompt_lookup_max}, "
            f"speculative_num_draft_tokens={server_args.speculative_num_draft_tokens}, "
            f"speculative_num_steps={server_args.speculative_num_steps}"
        )

        self._init_ngram_buffers()
        self.init_cuda_graphs()

    def _init_ngram_buffers(self):
        """Initialize buffers for n-gram matching kernel (shared by all code paths)."""

        # max_bs = self.target_worker.model_runner.max_running_requests
        # When get_batch_sizes_to_capture, if default max(capture_bs) > model_runner.req_to_token_pool.size,
        # capture_bs will align to model_runner.req_to_token_pool.size, and req_to_token_pool.size=max_running_request + 1.
        # (See model_runner.py for ReqToTokenPool creation logic)
        max_bs = self.target_worker.model_runner.req_to_token_pool.size
        draft_token_num = self.server_args.speculative_num_draft_tokens
        max_context_len = self.target_worker.model_runner.model_config.context_len

        with torch.device(self.device):
            self.context_tokens = torch.zeros(
                (max_bs, max_context_len), dtype=torch.int32
            )
            self.context_lens = torch.zeros((max_bs,), dtype=torch.int32)
            self.accept_lengths_buffer = torch.zeros((max_bs,), dtype=torch.int32)
            self.verified_tokens_buffer = torch.zeros(
                (max_bs * draft_token_num,), dtype=torch.int32
            )
            self.draft_tokens_output = torch.zeros(
                (max_bs, draft_token_num - 1), dtype=torch.int32
            )
            self.ngram_min_n = self.server_args.prompt_lookup_min
            self.ngram_max_n = self.server_args.prompt_lookup_max
            fixed_draft_len = draft_token_num - 1

            self.scores_list_buffer = [
                torch.ones((max_bs, 1, 1), dtype=torch.float32)
                for _ in range(fixed_draft_len)
            ]

            self.parents_list_buffer = []
            step0_parents = torch.full((max_bs, 2), -1, dtype=torch.long)
            step0_parents[:, 1] = 0
            self.parents_list_buffer.append(step0_parents)
            for step in range(1, fixed_draft_len):
                self.parents_list_buffer.append(
                    torch.full((max_bs, 1), step, dtype=torch.long)
                )            

    def init_cuda_graphs(self):
        self.cuda_graph_runner = None

        if self.server_args.disable_cuda_graph:
            return

        tic = time.time()
        logger.info("Capture cuda graph begin. This can take up to several minutes.")
        before_capture_available_gpu_memory = get_available_gpu_memory(
            self.device, self.target_worker.model_runner.gpu_id
        )
        logger.info(
            "Capture cuda graph begin. This can take up to several minutes. "
            f"avail mem={before_capture_available_gpu_memory:.2f} GB in model runner!"
        )

        from sglang.srt.speculative.pld_cuda_graph_runner import PLDCudaGraphRunner
        self.cuda_graph_runner = PLDCudaGraphRunner(self)

        after_capture_available_gpu_memory = get_available_gpu_memory(
            self.device, self.target_worker.model_runner.gpu_id
        )
        logger.info(
            f"Capture cuda graph end. Time elapsed: {time.time() - tic:.2f} s. "
            f"avail mem={after_capture_available_gpu_memory:.2f} GB"
        )
        logger.info(
            f"{len(self.cuda_graph_runner.graphs)} graphs used "
            f"mem={(before_capture_available_gpu_memory - after_capture_available_gpu_memory):.2f} GB"
        )

    def forward_batch_speculative_generation(
        self, model_worker_batch: ModelWorkerBatch, launch_done=None
    ):
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.target_worker.model_runner
        )

        vocab_masks = None
        if self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch):
            if forward_batch.forward_mode.is_target_verify():
                vocab_masks = self.preprocess_for_verify(forward_batch)
            out = self.cuda_graph_runner.replay(forward_batch, vocab_masks)
            self.target_worker.model_runner.req_to_token_pool.verified_lens[0].zero_()
        elif forward_batch.forward_mode.is_target_verify():
            vocab_masks = self.preprocess_for_verify(forward_batch)
            self.init_attn_backends(forward_batch)
            out = self.forward_decode_spec(forward_batch, vocab_masks)
        elif forward_batch.forward_mode.is_extend():
            self.init_attn_backends(forward_batch)
            out = self.forward_prefill_spec(model_worker_batch, forward_batch)
        elif forward_batch.forward_mode.is_idle():
            out = self.forward_idle(forward_batch)
        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")

        if launch_done:
            launch_done.set()
        return out

    def forward_decode_spec(
        self, forward_batch: ForwardBatch, vocab_masks: Optional[torch.Tensor] = None
    ):
        if self.use_over_embedding:
            forward_batch.oe_column_starts[:forward_batch.batch_size] = (
                forward_batch.req_to_token_pool.verified_lens[forward_batch.req_pool_indices]
            )
            forward_batch.oe_req_lens[:forward_batch.batch_size] = (
                self.server_args.speculative_num_draft_tokens
            )
            bs = forward_batch.batch_size
            num_verify_tokens = self.server_args.speculative_num_draft_tokens
            forward_batch.oe_out_column_starts[:bs] = (
                forward_batch.req_to_token_pool.verified_lens[forward_batch.req_pool_indices]
            )
            forward_batch.oe_out_req_lens[:bs] = num_verify_tokens
            update_token_table(
                oe_token_table=forward_batch.oe_token_table,
                tokens=forward_batch.input_ids.to(torch.int32),
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.oe_out_column_starts,
                oe_req_lens=forward_batch.oe_out_req_lens,
            )
        logits_output = self.target_worker.model_runner.forward_extend(
            forward_batch, skip_metadata_init=True
        )
        target_predict, logits_output, accept_length, accept_index = (
            self.rejection_sampling(forward_batch, logits_output, vocab_masks)
        )
        if self.use_over_embedding:
            bs = forward_batch.batch_size
            forward_batch.oe_out_column_starts[:bs] = (
                forward_batch.req_to_token_pool.verified_lens[forward_batch.req_pool_indices]
            )
            forward_batch.oe_out_req_lens[:bs] = accept_length

            update_token_table(
                oe_token_table=forward_batch.oe_token_table,
                tokens=target_predict,
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.oe_out_column_starts,
                oe_req_lens=forward_batch.oe_out_req_lens,
            )
        new_verified_id = self.preprocess_for_draft_after_decode(
            forward_batch, accept_length, accept_index, target_predict
        )

        output_ids = target_predict[accept_index]
        
        req_pool_indices = forward_batch.req_pool_indices
        self.target_worker.model_runner.req_to_token_pool.verified_lens[req_pool_indices] \
            += accept_length

        _scores_list, token_list, _parents_list = self.propose_with_ngram_kernel(
            forward_batch, target_predict, accept_length
        )
        return (
            logits_output,
            output_ids,
            accept_length,
            new_verified_id,
            token_list,
        )

    def forward_prefill_spec(
        self, model_worker_batch: ModelWorkerBatch, forward_batch: ForwardBatch
    ):
        """Handle prefill phase."""
        if self.use_over_embedding:
            forward_batch.oe_column_starts[:forward_batch.batch_size] = (
                forward_batch.extend_prefix_lens
            )
            forward_batch.oe_req_lens[:forward_batch.batch_size] = (
                forward_batch.extend_seq_lens
            )
        target_logits_output = self.target_worker.model_runner.forward_extend(
            forward_batch, skip_metadata_init=True
        )

        next_token_ids = self.target_worker.model_runner.sample(
            target_logits_output, forward_batch
        )

        if model_worker_batch.disagg_set_aux_fn is not None:
            model_worker_batch.disagg_set_aux_fn(next_token_ids, target_logits_output)

        bs = forward_batch.batch_size
        # TODO: _copy_context_to_buffers has performance overhead in large batches, consider optimizing
        self._copy_context_to_buffers(forward_batch, bs)

        req_pool_indices = forward_batch.req_pool_indices
        self.target_worker.model_runner.req_to_token_pool.verified_lens[req_pool_indices] \
            += forward_batch.new_tokens_to_compute
        accept_lengths_for_kernel = torch.ones(bs, dtype=torch.int32, device=self.device)

        # In prefill, next_token_ids is 1D [bs], but _ngram_matching_kernel expects
        # verified_tokens to be in the format that can be indexed by [bs * draft_token_num]
        # For prefill, we only have 1 verified token per request, so we need to reshape it
        # to match the expected format: [token1, pad, pad, ..., token2, pad, pad, ...]
        # where each request occupies draft_token_num slots
        draft_token_num = self.server_args.speculative_num_draft_tokens
        verified_tokens_for_kernel = torch.zeros(
            bs * draft_token_num, dtype=next_token_ids.dtype, device=next_token_ids.device
        )
        # Place each next_token_id at the start of its draft_token_num-sized slot
        for i in range(bs):
            verified_tokens_for_kernel[i * draft_token_num] = next_token_ids[i]

        _scores_list, token_list, _parents_list = self.propose_with_ngram_kernel(
            forward_batch, verified_tokens_for_kernel, accept_lengths_for_kernel
        )

        return (
            target_logits_output,
            next_token_ids,
            None,  # accept_length (not used in prefill)
            next_token_ids,  # new_verified_id (same as next_token_ids in prefill)
            token_list,
        )

    def forward_idle(self, forward_batch: ForwardBatch):
        logits_output = self.target_worker.model_runner.forward(forward_batch)
        next_token_ids = self.target_worker.model_runner.sample(
            logits_output, forward_batch
        )

        from sglang.srt.speculative.eagle_utils import EagleDraftOutput
        forward_batch.spec_info = EagleDraftOutput(
            last_verified_ids=next_token_ids,
            score_list=[],
            token_list=[],
            parents_list=[],
        )

        return None, None, None, None, None

    def propose_with_ngram_kernel(
        self, forward_batch: ForwardBatch, verified_tokens: torch.Tensor, accept_lengths: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Unified n-gram matching using CUDA kernel.
        Returns EAGLE-compatible format: (scores_list, token_list, parents_list)
        """
        bs = forward_batch.batch_size

        self._ngram_matching_kernel(bs, verified_tokens, accept_lengths)

        token_list = self._construct_token_list(bs)

        # Return pre-computed scores and parents
        scores_list = [self.scores_list_buffer[i][:bs] for i in range(len(self.scores_list_buffer))]
        parents_list = [self.parents_list_buffer[i][:bs] for i in range(len(self.parents_list_buffer))]

        return scores_list, token_list, parents_list

    def _copy_context_to_buffers(self, forward_batch: ForwardBatch, bs: int):
        """
        Copy context information (origin_input_ids + output_ids) to buffers.
        This prepares data for n-gram matching kernel.
        """
        for i, req in enumerate(forward_batch.reqs):
            # Build context: origin_input_ids + output_ids
            context = req.origin_input_ids + req.output_ids
            context_len = len(context)

            if context_len > 0:
                max_len = min(context_len, self.context_tokens.shape[1])
                self.context_tokens[i, :max_len] = torch.tensor(
                    context[:max_len], dtype=torch.int32, device=self.context_tokens.device
                )
                self.context_lens[i] = max_len
            else:
                self.context_lens[i] = 0

    def _ngram_matching_kernel(
        self,
        bs: int,
        verified_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ):
        """
        N-gram matching CUDA kernel.

        This kernel will:
        1. Update context_tokens and context_lens based on verified tokens
        2. Perform n-gram matching to find draft tokens
        3. Write results to draft_tokens_output

        Args:
            bs: Batch size
            verified_tokens: Verified token IDs from target model
            accept_lengths: Number of accepted tokens per sequence [bs]

        Kernel inputs (from self):
            - context_tokens: [max_bs, max_context_len] - historical tokens
            - context_lens: [max_bs] - length of each context
            - ngram_min_n, ngram_max_n: n-gram matching parameters

        Kernel outputs (to self):
            - draft_tokens_output: [max_bs, draft_token_num-1] - proposed draft tokens
        """
        self.accept_lengths_buffer[:bs].copy_(accept_lengths)

        draft_token_num = self.server_args.speculative_num_draft_tokens
        max_context_len = self.context_tokens.shape[1]
        ngram_min_n = self.ngram_min_n
        ngram_max_n = self.ngram_max_n
        # Verified_tokens is 1D
        num_tokens = min(len(verified_tokens), bs * draft_token_num)
        self.verified_tokens_buffer[:num_tokens].copy_(verified_tokens[:num_tokens])

        ngram_matching(
            self.context_tokens,
            self.context_lens,
            self.verified_tokens_buffer,
            self.accept_lengths_buffer,
            self.draft_tokens_output,
            ngram_min_n,
            ngram_max_n,
            bs,
            draft_token_num,
            max_context_len,
        )

    def _construct_token_list(self, bs: int):
        """
        Construct token_list from draft_tokens_output.

        This converts the 2D draft_tokens_output tensor into the EAGLE-compatible
        list format expected by the rest of the system.

        Args:
            bs: Batch size

        Returns:
            token_list: List of tensors, each [bs, 1]
        """
        fixed_draft_len = self.server_args.speculative_num_draft_tokens - 1
        token_list = []

        for step in range(fixed_draft_len):
            tokens = self.draft_tokens_output[:bs, step].view(bs, 1)
            token_list.append(tokens)

        return token_list

    def preprocess_for_verify(self, forward_batch: ForwardBatch):
        from sglang.srt.speculative.eagle_utils import EagleVerifyInput, EagleDraftOutput

        assert isinstance(forward_batch.spec_info, EagleDraftOutput)

        bs = forward_batch.batch_size
        self._copy_context_to_buffers(forward_batch, bs)

        eagle_spec_info = forward_batch.spec_info
        forward_batch.seq_lens = self.target_worker.model_runner.req_to_token_pool.verified_lens[
            forward_batch.req_pool_indices
        ]
        forward_batch.extend_seq_lens = forward_batch.new_tokens_to_compute

        token_list = eagle_spec_info.token_list
        if isinstance(token_list, list):
            token_list = torch.cat(token_list, dim=1)
        verify_spec_info = EagleVerifyInput.create(
            verified_id=eagle_spec_info.last_verified_ids,
            token_list=token_list,
            seq_lens=forward_batch.seq_lens,
            spec_steps=self.speculative_num_steps,
            num_verify_tokens=self.server_args.speculative_num_draft_tokens,
            is_all_greedy=forward_batch.sampling_info.is_all_greedy,
            is_idle=False,
        )

        vocab_masks = None
        if forward_batch.sampling_info.grammars:
            grammars = forward_batch.sampling_info.grammars
            if forward_batch.sampling_info.sampling_info_done:
                forward_batch.sampling_info.sampling_info_done.wait()
            retrive_next_sibling = torch.full(
                (forward_batch.batch_size, self.speculative_num_steps + 1),
                -1,
                device="cpu",
                dtype=torch.long,
            )
            retrive_next_token = torch.full(
                (forward_batch.batch_size, self.speculative_num_steps + 1),
                -1,
                device="cpu",
                dtype=torch.long,
            )
            vocab_masks = generate_token_bitmask(
                grammars,
                verify_spec_info,
                retrive_next_token,
                retrive_next_sibling,
                verify_spec_info.draft_token.cpu(),
                forward_batch.sampling_info.vocab_size,
            )

            if vocab_masks is not None:
                assert verify_spec_info.grammar is not None
                vocab_masks = vocab_masks.to(verify_spec_info.draft_token.device)
                forward_batch.sampling_info.vocab_mask = None

        forward_batch.spec_info = verify_spec_info
        forward_batch.input_ids = verify_spec_info.draft_token
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch.positions = forward_batch.spec_info.positions
        return vocab_masks

    def init_attn_backends(self, forward_batch: ForwardBatch):
        self.target_worker.model_runner.attn_backend.init_forward_metadata(
            forward_batch
        )

    def rejection_sampling(
        self,
        forward_batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
        vocab_masks: Optional[torch.Tensor] = None,
    ):
        forward_batch.spec_info.hidden_states = logits_output.hidden_states
        predict, logits_output, accept_length, accept_index = (
            forward_batch.spec_info.verify(forward_batch, logits_output, vocab_masks)
        )
        return predict, logits_output, accept_length, accept_index

    def preprocess_for_draft_after_decode(
        self,
        forward_batch: ForwardBatch,
        accept_length: torch.Tensor,
        accept_index: torch.Tensor,
        target_predict: torch.Tensor,
    ):
        forward_batch.forward_mode = ForwardMode.DRAFT_EXTEND
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch.extend_seq_lens = accept_length
        forward_batch.token_to_kv_pool = self.target_worker.model_runner.token_to_kv_pool
        draft_input = EagleDraftInput()

        draft_input.hidden_states = forward_batch.spec_info.hidden_states
        draft_input.accept_length = accept_length
        draft_input.verified_id = target_predict
        draft_input.draft_token_num = self.server_args.speculative_num_draft_tokens
        draft_input.accept_index = accept_index
        new_verified_id = draft_input.prepare_extend_after_decode(
            forward_batch, self.use_over_embedding
        )
        forward_batch.spec_info = draft_input
        return new_verified_id

