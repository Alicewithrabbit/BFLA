import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from bfla_vllm.requests import current_requests, install_v1, install_v2


class RequestContexts(unittest.TestCase):
    def tearDown(self):
        self.assertIsNone(current_requests.get())

    def test_v1_context_tracks_full_prompt_and_clears_after_failure(self):
        class Runner:
            input_batch = NS(req_ids=['request-a'], num_prompt_tokens_cpu_tensor=torch.tensor([8192]))
            requests = {'request-a': object()}
            def _build_attention_metadata(self, n, b, q, **kwargs):
                c = current_requests.get()
                if kwargs.get('for_cudagraph_capture'): return c
                if kwargs.get('fail'): raise RuntimeError('test builder failure')
                return c.keys, c.prompt_lengths
        install_v1(Runner)
        # The boundary under test is CPU request propagation; pool cleanup is
        # exercised separately using its real dictionary below.
        with patch('bfla_vllm.execution.retain_requests', create=True):
            self.assertEqual(Runner()._build_attention_metadata(256,1,256), (('request-a',),(8192,)))
            self.assertIsNone(Runner()._build_attention_metadata(256,1,256,for_cudagraph_capture=True))
            with self.assertRaises(RuntimeError): Runner()._build_attention_metadata(256,1,256,fail=True)

    def test_v1_positional_capture_flag_follows_runner_signature(self):
        class Runner:
            def _build_attention_metadata(self, n, b, q, extra, for_cudagraph_capture=False):
                return current_requests.get()
        install_v1(Runner)
        # Capturing has no live request batch to read.
        self.assertIsNone(Runner()._build_attention_metadata(8, 1, 8, None, True))

    def test_v2_context_and_real_state_release(self):
        from bfla_vllm import execution
        class State:
            def prepare_attn(self,*args,**kwargs): return current_requests.get()
            def remove_request(self,request_id): return request_id
            def add_request(self,idx,data): return idx
        install_v2(State)
        state = State()
        state.add_request(0,NS(req_id='a',prompt_len=8192))
        state.add_request(1,NS(req_id='b',prompt_len=16384))
        # Resumed prefill_len includes generated history; it must not move tail.
        batch = NS(num_reqs=2,req_ids=['a','b'],prefill_len_np=[8300,16400])
        c = state.prepare_attn(batch,None,None,None,None,None)
        self.assertEqual(c.keys,('a','b'))
        self.assertEqual(c.prompt_lengths,(8192,16384))
        execution.POOLS[('layer','a')] = object()
        execution.POOLS[('layer','b')] = object()
        self.assertEqual(state.remove_request('a'),'a')
        self.assertNotIn(('layer','a'),execution.POOLS)
        self.assertIn(('layer','b'),execution.POOLS)
        execution.POOLS.clear()

    def test_v1_native_final_chunks_retire_only_completed_pools_after_forward(self):
        from bfla_vllm import execution
        from unittest.mock import patch

        class Runner:
            input_batch = NS(req_ids=['done', 'partial', 'decode', 'draft'],
                             num_computed_tokens_cpu=[80, 48, 128, 90],
                             num_prompt_tokens=[96, 96, 128, 96])
            def _build_attention_metadata(self, *args, **kwargs): pass
            def execute_model(self, scheduler_output, fail=False):
                self.during = set(execution.POOLS)
                if fail:
                    raise RuntimeError('incomplete forward')
                return 'native'

        install_v1(Runner)
        runner = Runner()
        pools = {(layer, req): object() for req in runner.input_batch.req_ids for layer in (1, 2)}
        original = set(pools)
        step = NS(num_scheduled_tokens={'done': 16, 'partial': 16, 'decode': 1, 'draft': 8},
                  scheduled_spec_decode_tokens={'draft': [1, 2, 3]})
        with patch.object(execution, 'POOLS', pools):
            with self.assertRaisesRegex(RuntimeError, 'incomplete forward'):
                runner.execute_model(step, fail=True)
            self.assertEqual(set(pools), original)
            self.assertEqual(runner.execute_model(step), 'native')
            self.assertEqual(runner.during, original)
            self.assertEqual(set(pools), {k for k in original if k[1] != 'done'})

    def test_final_prefills_retire_in_one_pool_scan_on_both_runners(self):
        from unittest.mock import patch
        from bfla_vllm import execution
        from bfla_vllm.requests import install_v2_runner

        class Pools(dict):
            scans = 0
            def __iter__(self):
                self.scans += 1
                return super().__iter__()

        ids = ('a', 'b', 'partial', 'draft', 'decode')
        prompts = [10] * 5
        prefixes = [8, 6, 3, 8, 10]
        counts = [2, 4, 1, 2, 1]
        step = NS(num_scheduled_tokens=dict(zip(ids, counts)), scheduled_spec_decode_tokens={'draft': [1]})

        class V1:
            input_batch = NS(req_ids=ids, num_prompt_tokens=prompts, num_computed_tokens_cpu=prefixes)
            def _build_attention_metadata(self, *args): pass
            def execute_model(self, *args): pass
        class V2:
            model_state = NS(_bfla_prompt_lengths=dict(zip(ids, prompts)))
            execute_model_state = None
            def gather_batch_req_state(self, *args): pass
            def execute_model(self, *args):
                self.execute_model_state = NS(input_batch=NS(has_prefill=True, num_reqs=5, req_ids=ids,
                    is_prefilling_np=[True]*4+[False], num_computed_prefill_tokens_np=prefixes,
                    num_scheduled_tokens=counts, num_draft_tokens_per_req=[0, 0, 0, 1, 0]))
        install_v1(V1)
        install_v2_runner(V2)
        for runner in (V1(), V2()):
            with self.subTest(runner=type(runner).__name__):
                pools = Pools({(layer, req): object() for req in ids for layer in (1, 2)})
                with patch.object(execution, 'POOLS', pools):
                    runner.execute_model(step)
                self.assertEqual(set(pools.keys()), {(l, k) for l in (1, 2) for k in ids[2:]})
                self.assertEqual(pools.scans, 1)

    def test_layer_completion_leaves_other_layers_and_unfinished_rows(self):
        from unittest.mock import patch
        from bfla_vllm import execution
        pools = {(layer, row, cfg): object() for layer in ('first','last')
                 for row in ('a','b','partial') for cfg in (1, 2)}
        with patch.object(execution, 'POOLS', pools):
            execution.release_requests({'a', 'b'}, layer_owner='first')
        self.assertEqual(set(pools), {(l, r, c) for l in ('first','last') for r in ('a','b','partial')
                                      for c in (1, 2) if l == 'last' or r == 'partial'})

    def test_v2_finish_notifications_batch_successful_removals(self):
        from unittest.mock import patch
        from bfla_vllm import execution
        from bfla_vllm.requests import install_v2, install_v2_runner

        class State:
            def prepare_attn(self, *args): pass
            def add_request(self, *args): pass
            def remove_request(self, key):
                if key == 'failed':
                    raise RuntimeError(key)
        class Runner:
            def execute_model(self, *args): pass
            def gather_batch_req_state(self, *args): pass
            def finish_requests(self, step):
                for key in sorted(step.finished_req_ids | step.preempted_req_ids):
                    self.model_state.remove_request(key)
        class Pools(dict):
            scans = 0
            def __iter__(self):
                self.scans += 1
                return super().__iter__()
        install_v2(State)
        install_v2_runner(Runner)
        runner = Runner()
        runner.model_state = State()
        pools = Pools({(0, k): object() for k in ('a','b','failed','keep')})
        with patch.object(execution, 'POOLS', pools):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                runner.finish_requests(NS(finished_req_ids={'a','failed'}, preempted_req_ids={'b'}))
            self.assertEqual(set(pools.keys()), {(0,'failed'), (0,'keep')})
            self.assertEqual(pools.scans, 1)
            runner.model_state.remove_request('keep')
            self.assertNotIn((0,'keep'), pools)

    def test_v2_admission_batches_invalidations_even_on_partial_failure(self):
        from bfla_vllm import execution
        from bfla_vllm.requests import install_v2_retirement

        class State:
            def prepare_attn(self, *args): pass
            def remove_request(self, key): pass
            def add_request(self, idx, data):
                if data.req_id == 'fails':
                    raise ValueError('admission failed')

        class Hybrid(State):
            def prepare_attn(self, *args): pass
            def add_request(self, idx, data):
                return super().add_request(idx, data)

        class Runner:
            def execute_model(self, *args): pass
            def add_requests(self, step):
                for idx, data in enumerate(step.scheduled_new_reqs):
                    self.model_state.remove_request(data.req_id)
                    self.model_state.add_request(idx, data)

        class Pools(dict):
            scans = 0
            def __iter__(self):
                self.scans += 1
                return super().__iter__()

        install_v2(State)
        install_v2(Hybrid)
        install_v2_retirement(Runner)
        for state in (State(), Hybrid()):
            with self.subTest(state=type(state).__name__):
                runner = Runner()
                runner.model_state = state
                keys = ('readded', 'fresh', 'fails', 'pending', 'other')
                pools = Pools({(layer, k): object() for layer in (0, 1) for k in keys})
                with patch.object(execution, 'POOLS', pools):
                    requests = [NS(req_id=k, prompt_len=10) for k in keys[:-1]]
                    with self.assertRaisesRegex(ValueError, 'admission failed'):
                        runner.add_requests(NS(scheduled_new_reqs=requests))
                    self.assertEqual(pools.scans, 1)
                    self.assertEqual(set(pools.keys()), {(l, k) for l in (0, 1) for k in keys[3:]})
                    self.assertEqual(state._bfla_prompt_lengths, {'readded': 10, 'fresh': 10})
                    runner.add_requests(NS(scheduled_new_reqs=[]))
                    self.assertEqual(pools.scans, 1)
                    runner.add_requests(NS(scheduled_new_reqs=[requests[-1]]))
                    self.assertEqual(pools.scans, 2)
                    self.assertEqual(set(pools.keys()), {(0, 'other'), (1, 'other')})

    def test_v1_resumed_request_invalidates_pool_even_if_id_is_retained(self):
        from bfla_vllm import execution
        class Runner:
            def _build_attention_metadata(self,*args,**kwargs): pass
            def _update_states(self,scheduler_output): return 'native-update'
        install_v1(Runner)
        execution.POOLS[('layer','resumed')] = object()
        execution.POOLS[('layer','unchanged')] = object()
        scheduler = NS(finished_req_ids=set(),preempted_req_ids=set(),scheduled_new_reqs=[],
                       scheduled_cached_reqs=NS(resumed_req_ids={'resumed'}))
        self.assertEqual(Runner()._update_states(scheduler),'native-update')
        self.assertNotIn(('layer','resumed'),execution.POOLS)
        self.assertIn(('layer','unchanged'),execution.POOLS)
        execution.POOLS.clear()


if __name__ == '__main__': unittest.main()
