import os
import time
import triton
from typing import Optional

import torch
from huggingface_hub import snapshot_download

from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    EagleDraftOutput,
    EagleVerifyInput,
    fast_topk,
    generate_token_bitmask,
    prepare_for_multi_step_draft_kernel,
    update_oe_metadata,
    update_draft_decode_cache
)
from sglang.srt.speculative.spec_decoding_cuda_graph_runner import (
    SpecDecodeCudaGraphRunner,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.oe_utils import update_token_table
from sglang.srt.utils import get_available_gpu_memory, get_colorful_logger

from sglang.srt.configs.model_config import AttentionArch

from flashinfer.sampling import softmax

logger = get_colorful_logger(__name__)


class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        attn_tp_rank: int,
        moe_ep_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
        global_rank: int,
    ):
        # Do not capture cuda graph in `super().__init__()`
        # We will capture it later
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        if server_args.speculative_token_map is not None:
            if os.path.exists(server_args.speculative_token_map):
                self.hot_token_id = torch.load(server_args.speculative_token_map)
            else:
                cache_dir = snapshot_download(
                    os.path.dirname(server_args.speculative_token_map),
                    ignore_patterns=["*.bin", "*.safetensors"],
                )
                file_path = os.path.join(
                    cache_dir, os.path.basename(server_args.speculative_token_map)
                )
                self.hot_token_id = torch.load(file_path)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )

        # Share req_to_token_pool and kv_allocator with target worker
        self.target_worker = target_worker
        self.req_to_token_pool = self.target_worker.model_runner.req_to_token_pool
        self.kv_allocator = self.target_worker.model_runner.kv_allocator
        self.oe_token_table = self.target_worker.model_runner.oe_token_table

        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            attn_tp_rank=attn_tp_rank,
            moe_ep_rank=moe_ep_rank,
            global_rank=global_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=self.req_to_token_pool,
            kv_allocator=self.kv_allocator,
            oe_token_table=self.oe_token_table
        )

        # Parse arguments
        self.topk = server_args.speculative_eagle_topk
        assert self.topk == 1, "Tree Attention is abandoned for now."
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.server_args = server_args

        # Share the embedding and lm_head
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            head = head.clone()
            self.hot_token_id = torch.tensor(
                self.hot_token_id, dtype=torch.int32, device=head.device
            )
            head.data = head.data[self.hot_token_id]
        else:
            self.hot_token_id = None

        self.use_over_embedding = self.use_over_embedding or self.target_worker.use_over_embedding

        if self.speculative_algorithm.is_eagle3():
            if self.target_worker.use_over_embedding:
                word_embed = embed.word_embeder.weight
            else:
                word_embed = embed
            self.model_runner.model.set_embed(word_embed)
            self.hot_token_id = self.model_runner.model.get_hot_token_id().to(
                word_embed.device
            )
        else:
            if self.use_over_embedding:
                self.model_runner.model.set_oe_and_head(embed, head)
            else:
                self.model_runner.model.set_embed_and_head(embed, head)
        self.model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph

        # Create multi-step attn backends and cuda graph runners
        drafter_backend = server_args.drafter_attention_backend
        self.drafter_backend = drafter_backend
        if drafter_backend == "flashinfer":
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackendForTVDGraph
            )
            self.draft_attn_backend = FlashInferMultiStepDraftBackendForTVDGraph(
                self.model_runner,
                self.speculative_num_steps,
            )
        elif drafter_backend == "triton":
            from sglang.srt.layers.attention.triton_backend import (
                TritonMultiStepDraftBackend,
            )

            self.draft_attn_backend = TritonMultiStepDraftBackend(
                self.model_runner,
                self.topk,
                self.speculative_num_steps,
            )
        elif drafter_backend == "flashmla":
            from sglang.srt.layers.attention.flashmla_backend import (
                FlashMLAMultiStepDecodeBackend,
            )
            self.draft_attn_backend = FlashMLAMultiStepDecodeBackend(
                self.model_runner, self.speculative_num_steps
            )
        elif drafter_backend == "flashinfer_mla":
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )
            self.draft_attn_backend = FlashInferMLAMultiStepDraftBackend(
                self.model_runner,
                self.topk,
                self.speculative_num_steps,
            )
        elif drafter_backend == "duo_attn":
            if self.model_runner.model_config.attention_arch != AttentionArch.MLA:
                from sglang.srt.layers.attention.flashinfer_backend import (
                    FlashInferMultiStepDraftBackend,
                )
                self.draft_attn_backend = FlashInferMultiStepDraftBackend(
                    self.model_runner,
                    self.topk,
                    self.speculative_num_steps,
                )
            else:
                from sglang.srt.layers.attention.duo_attn_backend import (
                    DuoAttnMultiStepBackend,
                )
                self.draft_attn_backend = DuoAttnMultiStepBackend(
                    self.model_runner,
                    self.topk,
                    self.speculative_num_steps,
                )
        elif drafter_backend == "hybrid_linear_attn":
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            self.draft_attn_backend = FlashInferMultiStepDraftBackend(
                self.model_runner, self.topk, self.speculative_num_steps
            )
        elif drafter_backend == "dsa":
            from sglang.srt.layers.attention.dsa_backend import (
                DpskSparseAttnMultiStepBackend,
            )
            self.draft_attn_backend = DpskSparseAttnMultiStepBackend(self.model_runner, self.topk, self.speculative_num_steps)
        else:
            raise ValueError(
                f"EAGLE is not supported with drafter attention backend {drafter_backend}"
            )

        self.init_cuda_graphs()

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None

        if self.server_args.disable_cuda_graph:
            return

        tic = time.time()
        logger.info("Capture cuda graph begin. This can take up to several minutes.")
        before_capture_available_gpu_memory = get_available_gpu_memory(
            self.device, self.model_runner.gpu_id
        )
        logger.info(
            "Capture cuda graph begin. This can take up to several minutes. "
            f"avail mem={before_capture_available_gpu_memory:.2f} GB in model runner!"
        )
        self.cuda_graph_runner = SpecDecodeCudaGraphRunner(self)
        after_capture_available_gpu_memory = get_available_gpu_memory(
            self.device, self.model_runner.gpu_id
        )
        logger.info(
            f"Capture cuda graph end. Time elapsed: {time.time() - tic:.2f} s. "
            f"avail mem={after_capture_available_gpu_memory:.2f} GB"
        )
        logger.info(
            f"{len(self.cuda_graph_runner.graphs)} graphs used "
            f"mem={(before_capture_available_gpu_memory - after_capture_available_gpu_memory):.2f} GB"
        )

    def init_attn_backends(self, forward_batch: ForwardBatch):
        self.target_worker.model_runner.attn_backend.init_forward_metadata(
            forward_batch
        )
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

    def preprocess_for_verify(
        self, forward_batch: ForwardBatch
    ) -> Optional[torch.tensor]:
        assert isinstance(forward_batch.spec_info, EagleDraftOutput)
        forward_batch.seq_lens = self.req_to_token_pool.verified_lens[
            forward_batch.req_pool_indices
        ]
        forward_batch.extend_seq_lens = forward_batch.new_tokens_to_compute
        verify_spec_info = EagleVerifyInput.create(
            forward_batch.spec_info.last_verified_ids,
            forward_batch.spec_info.token_list,
            forward_batch.seq_lens,
            self.speculative_num_steps,
            self.server_args.speculative_num_draft_tokens,
            forward_batch.sampling_info.is_all_greedy,
            False,
        )
        vocab_masks = None
        if forward_batch.sampling_info.grammars:
            grammars = forward_batch.sampling_info.grammars
            if forward_batch.sampling_info.sampling_info_done:
                forward_batch.sampling_info.sampling_info_done.wait()
            retrive_next_sibling = torch.full(
                (forward_batch.batch_size, self.speculative_num_steps + 1), -1, device="cpu", dtype=torch.long
            )
            retrive_next_token = torch.full(
                (forward_batch.batch_size, self.speculative_num_steps + 1), -1, device="cpu", dtype=torch.long
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
                # otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                forward_batch.sampling_info.vocab_mask = None

        forward_batch.spec_info = verify_spec_info
        forward_batch.input_ids = forward_batch.spec_info.draft_token
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch.positions = forward_batch.spec_info.positions

        return vocab_masks

    def forward_target_verify(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_target_verify()
        forward_batch.attn_backend = self.target_worker.model_runner.attn_backend
        logits_output = self.target_worker.model_runner.forward_extend(
            forward_batch, skip_metadata_init=True
        )
        return logits_output

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
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool
        forward_batch.attn_backend = self.model_runner.attn_backend
        draft_input = EagleDraftInput()
        # rearranged hidden states
        draft_input.hidden_states = forward_batch.spec_info.hidden_states
        draft_input.accept_length = accept_length
        draft_input.verified_id = target_predict
        draft_input.draft_token_num = self.server_args.speculative_num_draft_tokens
        draft_input.accept_index = accept_index
        new_verified_id = draft_input.prepare_extend_after_decode(forward_batch, self.use_over_embedding)
        forward_batch.spec_info = draft_input
        return new_verified_id

    def forward_draft_extend(self, forward_batch: ForwardBatch):
        forward_batch.attn_backend = self.model_runner.attn_backend
        logits_output = self.model_runner.forward_extend(
            forward_batch, skip_metadata_init=True
        )
        self.capture_for_decode(logits_output, forward_batch)

    def prepare_for_multi_step_draft(
        self, forward_batch: ForwardBatch, accept_lengths: torch.Tensor
    ):
        bs = forward_batch.batch_size
        out_cache_loc_for_draft_decode = torch.empty(
            size=(bs * (self.speculative_num_steps - 1),),
            dtype=torch.int32,
            device=self.device
        )
        seq_lens = torch.empty(bs, dtype=torch.int32, device=self.device)
        seq_lens_sum = torch.empty(1, dtype=torch.int32, device=self.device)
        prepare_for_multi_step_draft_kernel[(bs,)](
            out_cache_loc_ptr=out_cache_loc_for_draft_decode,
            verified_lens_ptr=self.req_to_token_pool.verified_lens,
            req_pool_indices_ptr=forward_batch.req_pool_indices,
            accept_lengths_ptr=accept_lengths,
            seq_lens_ptr=seq_lens,
            seq_lens_sum_ptr=seq_lens_sum,
            req_to_token_ptr=self.req_to_token_pool.req_to_token,
            req_to_token_ptr_stride=self.req_to_token_pool.req_to_token.shape[1],
            spec_num_steps=self.speculative_num_steps,
            bs=bs,
            bs_upper=triton.next_power_of_2(bs),
        )
        if self.speculative_num_steps > 1:
            forward_batch.seq_lens = seq_lens
            forward_batch.seq_lens_sum = seq_lens_sum
            forward_batch.positions = seq_lens
            forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            forward_batch.forward_mode = ForwardMode.DECODE
            forward_batch.out_cache_loc = out_cache_loc_for_draft_decode

    def eagle_style_propose(
        self, forward_batch: ForwardBatch, accept_lengths: torch.Tensor
    ):
        forward_batch.attn_backend = self.model_runner.attn_backend

        self.forward_draft_extend(forward_batch)
        self.prepare_for_multi_step_draft(forward_batch, accept_lengths)
        token_list = self.draft(forward_batch)
        return token_list

    def prepare_for_draft_prefill(
        self,
        forward_batch: ForwardBatch,
        target_logits_output: LogitsProcessorOutput,
        next_token_ids: torch.Tensor,
    ):
        forward_batch.forward_mode = ForwardMode.EXTEND
        if self.use_over_embedding:
            forward_batch.oe_column_starts[:forward_batch.batch_size] = forward_batch.extend_prefix_lens + 1
            forward_batch.oe_req_lens[:forward_batch.batch_size] = forward_batch.extend_seq_lens
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch.spec_info = EagleDraftInput(
            hidden_states=target_logits_output.hidden_states,
            verified_id=next_token_ids,
        )
        forward_batch.spec_info.set_input_ids(forward_batch)
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool
        forward_batch.attn_backend = self.model_runner.attn_backend

    def forward_decode_spec(
        self, forward_batch: ForwardBatch, vocab_masks: Optional[torch.Tensor] = None
    ):
        if self.use_over_embedding:
            forward_batch.oe_column_starts[:forward_batch.batch_size] = forward_batch.req_to_token_pool.verified_lens[forward_batch.req_pool_indices]
            forward_batch.oe_req_lens[:forward_batch.batch_size] = self.server_args.speculative_num_draft_tokens
        logits_output = self.forward_target_verify(forward_batch)
        target_predict, logits_output, accept_length, accept_index = (
            self.rejection_sampling(forward_batch, logits_output, vocab_masks)
        )
        # Results from target_predict, verify_lens not updated yet, just write at current verify length + 1 and continue
        if self.use_over_embedding:
            forward_batch.oe_out_column_starts[:forward_batch.batch_size] = \
                self.req_to_token_pool.verified_lens[forward_batch.req_pool_indices] + 1
            forward_batch.oe_out_req_lens[:forward_batch.batch_size] = self.server_args.speculative_num_draft_tokens
            update_token_table(
                oe_token_table=forward_batch.oe_token_table,
                tokens=target_predict,
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.oe_out_column_starts,
                oe_req_lens=forward_batch.oe_out_req_lens
            )
        new_verified_id = self.preprocess_for_draft_after_decode(
            forward_batch, accept_length, accept_index, target_predict
        )
        token_list = self.eagle_style_propose(
            forward_batch, accept_length
        )
        output_ids = target_predict[accept_index]
        return (
            logits_output,
            output_ids,
            accept_length,
            new_verified_id,
            token_list
        )

    def forward_prefill_spec(
        self, model_worker_batch: ModelWorkerBatch, forward_batch: ForwardBatch
    ):
        if self.use_over_embedding:
            forward_batch.oe_column_starts[:forward_batch.batch_size] = forward_batch.extend_prefix_lens
            forward_batch.oe_req_lens[:forward_batch.batch_size] = forward_batch.extend_seq_lens
        target_logits_output = self.target_worker.model_runner.forward_extend(
            forward_batch, skip_metadata_init=True
        )
        next_token_ids = self.target_worker.model_runner.sample(
            target_logits_output, forward_batch
        )
        if model_worker_batch.disagg_set_aux_fn is not None:
            model_worker_batch.disagg_set_aux_fn(next_token_ids, target_logits_output)
        self.prepare_for_draft_prefill(
            forward_batch, target_logits_output, next_token_ids
        )
        token_list = self.eagle_style_propose(
            forward_batch, forward_batch.new_tokens_to_compute
        )
        return (
            target_logits_output,
            next_token_ids,
            None,
            next_token_ids,
            token_list,
        )

    def forward_idle(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_idle()
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        logits_output = self.target_worker.model_runner.forward(forward_batch)
        next_token_ids = self.target_worker.model_runner.sample(
            logits_output, forward_batch
        )
        forward_batch.spec_info = EagleDraftInput(
            hidden_states=logits_output.hidden_states,
            verified_id=next_token_ids,
        )
        #self.capture_for_decode(logits_output, forward_batch)
        self.model_runner.forward_idle(forward_batch)
        for _ in range(self.speculative_num_steps - 1):
            self.model_runner.forward_idle(forward_batch)
        return None, None, None, None, None

    def forward_batch_speculative_generation(
        self, model_worker_batch: ModelWorkerBatch, launch_done=None
    ):
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.target_worker.model_runner
        )
        vocab_masks = None
        if self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch):
            vocab_masks = None
            if forward_batch.forward_mode.is_target_verify():
                vocab_masks = self.preprocess_for_verify(forward_batch)
            out = self.cuda_graph_runner.replay(forward_batch, vocab_masks)
            # set padding position to zero in case of illegal memory
            # when cuda graph padding happens
            self.req_to_token_pool.verified_lens[0].zero_()
            if isinstance(forward_batch.attn_backend, HybridLinearAttnBackend):
                accept_length = out[2]
                forward_batch.attn_backend.update_mamba_state_after_mtp_verify(accept_length, None)
        elif forward_batch.forward_mode.is_target_verify():
            vocab_masks = self.preprocess_for_verify(forward_batch)
            self.init_attn_backends(forward_batch)
            out = self.forward_decode_spec(forward_batch, vocab_masks)
            if isinstance(forward_batch.attn_backend, HybridLinearAttnBackend):
                accept_length = out[2]
                forward_batch.attn_backend.update_mamba_state_after_mtp_verify(accept_length, None)
        elif forward_batch.forward_mode.is_extend():
            self.init_attn_backends(forward_batch)
            forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            out = self.forward_prefill_spec(model_worker_batch, forward_batch)
        elif forward_batch.forward_mode.is_idle():
            out = self.forward_idle(forward_batch)
        if launch_done:
            launch_done.set()
        return out

    def draft(self, forward_batch: ForwardBatch):
        # Initialize attention backend
        if not forward_batch.forward_mode.is_idle() and self.speculative_num_steps > 1:
            self.draft_attn_backend.init_forward_metadata(forward_batch)
        # Run forward steps
        token_list = self.draft_forward(forward_batch)
        return token_list
    
    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        # out_cache_loc here:
        # <-- req 1 --> <-- req 2 --> <-- req 3 --> .....
        # [step1, step2, step1, step2, step1, step2]
        # Need to select step-wise cache loc when doing multi-step decode
        out_cache_loc = forward_batch.out_cache_loc
        if self.server_args.enable_dp_attention:
            forward_batch.global_num_tokens = forward_batch.global_batch_size
        _, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        # Return values
        token_list: torch.Tensor = torch.empty((forward_batch.batch_size, self.server_args.speculative_num_steps), dtype=torch.int32, device="cuda")

        # Forward multiple steps
        for i in range(self.speculative_num_steps):
            input_ids = topk_index.flatten()
            if self.use_over_embedding:
                update_oe_metadata(forward_batch, i, self.speculative_num_steps)
                # OE needs to update token_table and corresponding table_column_starts and req_lens
                update_token_table(forward_batch.oe_token_table,
                                   input_ids.to(torch.int32),
                                   forward_batch.req_pool_indices,
                                   forward_batch.oe_out_column_starts,
                                   forward_batch.oe_out_req_lens)
            token_list[:, i] = input_ids

            # we don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids.to(torch.int32)
            update_draft_decode_cache(
                out_cache_loc=out_cache_loc,
                forward_batch=forward_batch,
                draft_decode_step=i,
                speculative_num_steps=self.speculative_num_steps
            )
            if self.drafter_backend == "flashinfer":
                forward_batch.attn_backend = self.draft_attn_backend
                forward_batch.attn_backend.set_draft_step(i)
            else:
                forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states
            logits_output = self.model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            forward_batch.positions.add_(1)
            probs = softmax(logits_output.next_token_logits)
            # Get topk tokens for next position
            _, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]

            # Update last hidden_states
            hidden_states = logits_output.hidden_states

        return token_list

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, forward_batch: ForwardBatch
    ):
        probs = softmax(logits_output.next_token_logits)
        spec_info = forward_batch.spec_info
        spec_info.topk_p, spec_info.topk_index = fast_topk(probs, self.topk, dim=-1)
        spec_info.hidden_states = logits_output.hidden_states
