import unittest
import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

class TestThinkingBudgets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def test_case_1(self):
        think_end_ids = torch.tensor([[110, 119, 120], [110, 119, 120]])
        thinking_budgets = torch.tensor([10, 20])
        seq_lens = torch.tensor([0, 0])
        last_token_ids = torch.full_like(think_end_ids, -1)
        sampling_info = SamplingBatchInfo(
            temperatures = None,
            top_ps=None,
            top_ks=None,
            min_ps=None,
            is_all_greedy=False,
            need_min_p_sampling=False,
            vocab_size=0,
            thinking_budgets=thinking_budgets,
            last_token_ids=last_token_ids,
            think_end_ids=think_end_ids)

        for i in range(1, 30):
            seq_lens += 1
            next_token_ids = torch.randint(100, (2,))
            sampling_info.apply_thinking_budgets(seq_lens, next_token_ids)
            self.assertTrue((sampling_info.last_token_ids[:,-1:].squeeze(-1) == next_token_ids).all())
            if i == 12:
                self.assertTrue((sampling_info.thinking_budgets == torch.tensor([-1, 20])).all())
            if i == 22:
                self.assertTrue((sampling_info.thinking_budgets == torch.tensor([-1, -1])).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
