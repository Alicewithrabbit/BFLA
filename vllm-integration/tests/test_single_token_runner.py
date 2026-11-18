"""Ordinary vLLM 0.31 runner phase, graph dispatch and pool retirement."""
import unittest
from collections import namedtuple
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from bfla_vllm import execution, requests


def gpu_runner_config(runner, tp_size=1):
    from bfla_vllm.backend import PrefillMixin
    impl = PrefillMixin()
    impl.attn_type = 'decoder'
    impl.sliding_window = (-1, -1)
    impl.kv_cache_dtype = 'auto'
    impl.head_size, impl.num_heads, impl.num_kv_heads = 256, 24 // tp_size, 4 // tp_size
    runner.parallel_config = NS(tensor_parallel_size=tp_size)
    runner.model_config = NS(hf_text_config=NS(num_attention_heads=24, num_key_value_heads=4))
    runner.compilation_config = NS(static_forward_context={'global': NS(impl=impl)})


class SingleTokenRunner(unittest.TestCase):
    def setUp(self):
        class ModelState:
            def add_request(self, index, data): pass
            def remove_request(self, req_id): pass
            def prepare_attn(self, *args):
                return requests.current_requests.get()

        class Runner:
            def gather_batch_req_state(self, scheduler_output, dummy_run):
                return (None, 1) if dummy_run else (self.batch, 1)

            def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
                self.gather_batch_req_state(scheduler_output, False)
                self.during_forward = tuple(execution.POOLS)
                self.execute_model_state = NS(input_batch=self.batch)
                return 'replayed'

        requests.install_v2(ModelState)
        self.state = ModelState()
        self.runner = Runner()
        gpu_runner_config(self.runner)
        self.runner.model_state = self.state
        self.runner.execute_model_state = None
        for i, key in enumerate(('prompt', 'decode')):
            self.state.add_request(i, NS(req_id=key, prompt_len=16384))
        requests.install_v2_runner(Runner)
        self.batch_type = namedtuple('Batch', 'req_ids num_reqs num_scheduled_tokens '
            'num_draft_tokens_np num_draft_tokens_per_req num_computed_prefill_tokens_np '
            'is_prefilling_np has_prefill prefill_runs_as_decode_np decode_graph_eligible')
        self.batch = self.batch_type(['prompt', 'decode'], 2, np.array([1, 1]), None, None,
            np.array([8192, 16384]), np.array([True, False]), True,
            np.array([True, False]), True)
        self.runner.batch = self.batch

    def test_existing_context_one_token_prefill_leaves_decode_graph(self):
        batch, uniform = self.runner.gather_batch_req_state(None, False)
        self.assertIsNone(uniform)
        self.assertFalse(batch.decode_graph_eligible)
        context = self.state.prepare_attn(batch, None, None, None, None, None)
        self.assertEqual(context.prefill_flags([False, False]), [True, False])
        self.assertEqual(batch.prefill_runs_as_decode_np.tolist(), [True, False])
        from bfla_vllm.backend import MetadataMixin

        class Native:
            def build(self, prefix, common, fast=False):
                return NS(query_start_loc=common.query_start_loc_cpu,
                          seq_lens=common.seq_lens_cpu_upper_bound,
                          block_table=torch.tensor([[1], [2]]), slot_mapping=torch.arange(2))

        class Builder(MetadataMixin, Native):
            vllm_config = NS(model_config=None, scheduler_config=NS(enable_chunked_prefill=True))

        common = NS(max_query_len=1, num_reqs=2, causal=True,
                    query_start_loc_cpu=torch.tensor([0, 1, 2]),
                    seq_lens_cpu_upper_bound=torch.tensor([8193, 16385]),
                    is_prefilling=torch.tensor([False, False]))
        token = requests.current_requests.set(context)
        self.addCleanup(requests.current_requests.reset, token)
        plan = Builder().build(0, common).bfla_plan
        self.assertEqual((plan.reason, plan.request_keys), ('mixed_prefill', ('prompt',)))

    def test_tp_fallback_keeps_decode_graph_and_rechecks_communicator(self):
        gpu_runner_config(self.runner, tp_size=2)
        device_comm = NS(pynccl_comm=None)
        group = NS(world_size=2, device_communicator=device_comm)
        with patch.dict('sys.modules', {'vllm.distributed': NS(get_tp_group=lambda: group)}):
            for comm, expected in ((None, 1), (NS(disabled=True), 1),
                                   (NS(disabled=False), None), (None, 1)):
                device_comm.pynccl_comm = comm
                batch, uniform = self.runner.gather_batch_req_state(None, False)
                self.assertEqual(uniform, expected)
                self.assertEqual(batch.decode_graph_eligible, expected == 1)

    def test_native_head_geometry_retains_uniform_graph(self):
        layer = self.runner.compilation_config.static_forward_context['global']
        layer.impl.head_size = 128
        _, uniform = self.runner.gather_batch_req_state(None, False)
        self.assertEqual(uniform, 1)

    def test_masked_speculative_prompt_and_capture_keep_native_dispatch(self):
        drafts = np.array([2, 0])
        self.runner.batch = self.batch._replace(num_scheduled_tokens=np.array([3, 1]),
            num_draft_tokens_np=drafts, num_draft_tokens_per_req=drafts)
        batch, uniform = self.runner.gather_batch_req_state(None, False)
        self.assertEqual(uniform, 1)
        self.assertEqual(self.runner.gather_batch_req_state(None, True), (None, 1))
        context = self.state.prepare_attn(batch, None, None, None, None, None)
        for native_flags in ([False, False], [True, False]):
            self.assertEqual(context.prefill_flags(native_flags), [False, False])
        self.assertIsNone(self.state.prepare_attn(batch, None, None, None, None, None, True))

    def test_native_final_prompt_retires_its_pool_after_graph_replay(self):
        self.runner.batch = self.batch._replace(num_computed_prefill_tokens_np=np.array([16383, 16384]))
        _, uniform = self.runner.gather_batch_req_state(None, False)
        self.assertEqual(uniform, 1)
        pools = {('layer', 'prompt'): object(), ('layer', 'decode'): object()}
        with patch.object(execution, 'POOLS', pools):
            self.assertEqual(self.runner.execute_model(None), 'replayed')
            self.assertIn(('layer', 'prompt'), self.runner.during_forward)
            self.assertEqual(list(pools), [('layer', 'decode')])
        with patch.dict('os.environ', {'BFLA_TAIL_TILES': '0'}):
            _, uniform = self.runner.gather_batch_req_state(None, False)
            self.assertIsNone(uniform)

    def test_decode_and_disabled_or_short_prompt_retain_graph(self):
        self.runner.batch = self.batch._replace(has_prefill=False)
        self.assertEqual(self.runner.gather_batch_req_state(None, False)[1], 1)
        self.runner.batch = self.batch._replace(num_computed_prefill_tokens_np=np.array([80, 16384]))
        self.assertEqual(self.runner.gather_batch_req_state(None, False)[1], 1)
        self.runner.batch = self.batch
        with patch.dict('os.environ', {'BFLA_ENABLED': '0'}):
            self.assertEqual(self.runner.gather_batch_req_state(None, False)[1], 1)


class V1GraphDispatch(unittest.TestCase):
    def test_sparse_prompt_graph_guard_keeps_native_dense_tails_and_capture(self):
        class Runner:
            uniform_decode_query_len = 1

            def _determine_batch_execution_and_padding(self, num_tokens, num_reqs,
                    num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn,
                    allow_microbatching=True, force_eager=False, force_uniform_decode=None):
                return force_uniform_decode

        runner = Runner()
        gpu_runner_config(runner)
        runner.input_batch = NS(req_ids=['d', 'p'], num_prompt_tokens=np.array([8, 16384]),
                                num_computed_tokens_cpu=np.array([10, 8192]))
        requests.install_v1_graphs(Runner)
        args = (2, 2, np.array([1, 1]), 1, False)
        self.assertIs(runner._determine_batch_execution_and_padding(*args), False)
        self.assertTrue(runner._determine_batch_execution_and_padding(*args, force_uniform_decode=True))
        gpu_runner_config(runner, tp_size=2)
        group = NS(world_size=2, device_communicator=NS(pynccl_comm=None))
        with patch.dict('sys.modules', {'vllm.distributed': NS(get_tp_group=lambda: group)}):
            self.assertIsNone(runner._determine_batch_execution_and_padding(*args))
            group.device_communicator.pynccl_comm = NS(disabled=False)
            self.assertIs(runner._determine_batch_execution_and_padding(*args), False)
        runner.input_batch.num_computed_tokens_cpu[1] = 16380
        self.assertIsNone(runner._determine_batch_execution_and_padding(*args))
        runner.input_batch.num_computed_tokens_cpu[1] = 16383
        pools = {('layer', 'p'): object(), ('layer', 'd'): object()}
        with patch.object(execution, 'POOLS', pools):
            self.assertIsNone(runner._determine_batch_execution_and_padding(*args))
            self.assertIn(('layer', 'p'), pools)
        runner.input_batch.num_computed_tokens_cpu[1] = 16384
        self.assertIsNone(runner._determine_batch_execution_and_padding(*args))


class V2MicrobatchContext(unittest.TestCase):
    def test_both_slices_are_native_even_when_first_slice_index_is_zero(self):
        class State:
            def prepare_attn(self, *args, **kwargs):
                return requests.current_requests.get(), requests.sparse_prefill_disabled.get()
            def add_request(self, idx, data): pass
            def remove_request(self, req_id): pass

        class UBatchRunner:
            def prepare(self, batch):
                return tuple(self.model_state.prepare_attn(batch, None, None, None, None, None,
                             ubatch_idx=i) for i in (0, 1))
        requests.install_v2(State)
        requests.install_v2_ubatches(UBatchRunner)
        runner = UBatchRunner()
        runner.model_state = State()
        runner.model_state.add_request(0, NS(req_id='p', prompt_len=64))
        batch = NS(num_reqs=1, req_ids=['p'])
        self.assertEqual(runner.prepare(batch), ((None, True), (None, True)))
        self.assertFalse(requests.sparse_prefill_disabled.get())


class V1MicrobatchContext(unittest.TestCase):
    def test_keyword_and_positional_microbatches_suppress_sparse_dispatch(self):
        class Runner:
            input_batch = NS(req_ids=['one', 'two'], num_prompt_tokens_cpu_tensor=torch.tensor([8, 9]))
            requests = {'one': None, 'two': None}
            def _build_attention_metadata(self, n, b, q, num_tokens_padded=None,
                    num_reqs_padded=None, ubatch_slices=None, for_cudagraph_capture=False):
                return requests.current_requests.get(), requests.sparse_prefill_disabled.get()
        requests.install_v1(Runner)
        runner = Runner()
        with patch.object(execution, 'retain_requests') as retain:
            self.assertEqual(runner._build_attention_metadata(8, 2, 4, ubatch_slices=[1, 2]), (None, True))
            self.assertEqual(runner._build_attention_metadata(8, 2, 4, None, None, [1, 2]), (None, True))
            self.assertEqual(retain.call_count, 2)
        self.assertFalse(requests.sparse_prefill_disabled.get())
        context, disabled = runner._build_attention_metadata(8, 2, 4)
        self.assertFalse(disabled)
        self.assertEqual(context.keys, ('one', 'two'))
