"""CPU coverage for the vLLM 0.31 V1 runner's attention-group reuse."""
from importlib.util import find_spec
import unittest

import torch


@unittest.skipUnless(find_spec('vllm'), 'vLLM runtime required')
class FlashMetadataRebind(unittest.TestCase):
    def test_each_group_uses_its_own_pages_and_slots_for_native_decode(self):
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
        from bfla_vllm.partition import native_metadata
        from bfla_vllm.protocol import PrefillPlan
        from bfla_vllm.v031 import FlashMetadataBuilder

        builder = object.__new__(FlashMetadataBuilder)
        table = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
        slots = torch.arange(100, 105)
        # Check the contiguous slice and the interleaved gather. Decode token
        # offsets in the slot mapping differ from request indices in the table.
        for rows, expected_pages, expected_slots in (
            (((2, 4, 5, 9),), [[130, 131]], [204]),
            (((0, 0, 1, 8), (2, 4, 5, 9)), [[110, 111], [130, 131]], [200, 204]),
        ):
            with self.subTest(rows=rows):
                first = FlashAttentionMetadata(
                    num_actual_tokens=5, max_query_len=3, query_start_loc=torch.tensor([0, 1, 4, 5]),
                    max_seq_len=9, seq_lens=torch.tensor([8, 6, 9]),
                    block_table=table, slot_mapping=slots, use_cascade=False, common_prefix_len=0,
                    cu_prefix_query_lens=None, prefix_kv_lens=None, suffix_kv_lens=None)
                first.bfla_plan = PrefillPlan(reason='mixed_prefill', native_rows=rows)
                first.bfla_native_metadata = native_metadata(first, rows)
                old_pages = first.bfla_native_metadata.block_table.clone()
                old_slots = first.bfla_native_metadata.slot_mapping.clone()

                second = builder.update_block_table(first, table + 100, slots + 100)
                self.assertEqual(second.bfla_native_metadata.block_table.tolist(), expected_pages)
                self.assertEqual(second.bfla_native_metadata.slot_mapping.tolist(), expected_slots)
                torch.testing.assert_close(first.bfla_native_metadata.block_table, old_pages)
                torch.testing.assert_close(first.bfla_native_metadata.slot_mapping, old_slots)
                torch.testing.assert_close(second.block_table, table + 100)
                torch.testing.assert_close(second.slot_mapping, slots + 100)
