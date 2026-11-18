import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from bfla_vllm import execution
from bfla_vllm.requests import install_v1, install_v2_retirement


class RepairLifecycle(unittest.TestCase):
    def test_both_runners_invalidate_all_layers_before_forward(self):
        class V1:
            def _build_attention_metadata(self, *args):
                pass

            def _update_states(self, step):
                return set(execution.POOLS)

        class V2:
            execute_model_state = None

            def execute_model(self, step, *args):
                return set(execution.POOLS)

        install_v1(V1)
        install_v2_retirement(V2)
        step = NS(finished_req_ids=set(), preempted_req_ids=set(), scheduled_new_reqs=[],
                  scheduled_cached_reqs=NS(resumed_req_ids=set()),
                  block_table_updates={'ordinary': ([12],)},
                  bfla_repaired_req_ids=('repaired',))
        old = {(layer, key): object() for layer in (1, 2)
               for key in ('ordinary', 'repaired', 'unaffected')}
        for forward in (V1()._update_states, V2().execute_model):
            with self.subTest(forward=forward), patch.object(execution, 'POOLS', old.copy()):
                self.assertEqual(forward(step), {owner for owner in old if owner[1] != 'repaired'})

    def test_v2_restages_only_rewound_rows_with_replay_tokens(self):
        class Runner:
            execute_model_state = None

            def update_requests(self, step):
                self.req_states.num_computed_tokens_np = step.starts

            def execute_model(self, step, *args):
                self.update_requests(step)

        install_v2_retirement(Runner)
        runner = Runner()
        device_count = Mock()
        runner.req_states = NS(req_id_to_index={'writer': 0, 'shared': 1, 'decode': 2, 'ordinary': 3},
                               num_computed_tokens=device_count, prompt_len=NS(np=[64, 8192, 64, 8192]),
                               prefill_len=NS(np=[8192, 8192, 128, 8192]))
        step = NS(starts=[4032, 4352, 128, 4096], bfla_rewound_req_ids=('writer', 'decode'),
                  bfla_repaired_req_ids=('writer', 'shared', 'decode'),
                  block_table_updates={'ordinary': ([19],)})
        runner.execute_model(step)
        device_count.stage_write_elem.assert_called_once_with(0, 4032)
        device_count.apply_write.assert_called_once()

        device_count.reset_mock()
        runner.execute_model(NS(starts=[4288, 4608, 129, 4352], block_table_updates=step.block_table_updates))
        device_count.stage_write_elem.assert_not_called()
        device_count.apply_write.assert_not_called()


class SchedulerRepairLifecycle(unittest.TestCase):
    def setUp(self):
        from bfla_vllm.kv_repair import _RepairWindow

        class Request:
            def __init__(self, key):
                self.request_id = key
                self.num_preemptions = 1
                self.num_tokens = 300
                self.num_computed_tokens = 192
                self.num_in_flight_tokens = 64
                self.num_stale_output_tokens = 0
                self.block_hashes = list(range(8))
                self.is_finished = lambda: False

        self.request = Request('writer')
        self.other = Request('independent')
        self.cache = Mock()
        self.manager = NS(coordinator=NS(cache_blocks=self.cache), block_pool=Mock())
        self.scheduler = NS(requests={'writer':self.request, 'independent':self.other},
                            kv_cache_manager=self.manager, block_size=64, hash_block_size=64,
                            defer_block_free=True, connector=NS(get_num_new_matched_tokens=Mock()))
        self.window = self.scheduler._bfla_kv_repair_window = _RepairWindow(self.scheduler)

    def test_only_matching_rpc_commits_retry_frontier(self):
        request = self.request
        w = self.window
        w.retry(self.scheduler, request, 1, 192, {'writer':64})
        frontier = NS(num_scheduled_tokens={'writer':64}, scheduled_spec_decode_tokens={})
        unrelated = NS(num_scheduled_tokens={'independent':64}, scheduled_spec_decode_tokens={})
        w.submitted(self.scheduler, frontier)
        w.receiving(self.scheduler, unrelated)
        w.prune(self.scheduler)
        w.cache_confirmed(request, 256)
        self.cache.assert_called_with(request, 0)
        w.cache_confirmed(self.other, 256)
        self.cache.assert_called_with(self.other, 256)
        w.receiving(self.scheduler, frontier)
        w.cache_confirmed(request, 256)
        self.cache.assert_called_with(request, 192)
        w.prune(self.scheduler)
        self.assertFalse(hasattr(self.scheduler, '_bfla_kv_repair_window'))
        self.assertIs(self.manager.coordinator.cache_blocks, self.cache)

    def test_second_fault_invalidates_prior_receipts_and_native_preemption_resets_ack(self):
        request = self.request
        w = self.window
        step = NS(num_scheduled_tokens={'writer':64}, scheduled_spec_decode_tokens={})
        w.retry(self.scheduler, request, 1, 256, {'writer':64})
        w.submitted(self.scheduler, step)
        w.retry(self.scheduler, request, 2, 192, {'writer':64})
        w.receiving(self.scheduler, step)
        self.assertEqual(w.retries[request].acknowledged, 0)
        self.assertEqual(w.retries[request].target, 256)
        self.assertEqual(w.retries[request].first_hash, (1, 1))
        w.submitted(self.scheduler, step)
        w.receiving(self.scheduler, step)
        self.assertEqual(w.retries[request].acknowledged, 192)
        request.num_preemptions += 1
        w.cache_confirmed(request, 256)
        self.cache.assert_called_with(request, 0)


class RetiredAllocationRecovery(unittest.TestCase):
    def test_fault_propagates_through_retained_finished_and_preempted_tables(self):
        from bfla_vllm.kv_repair import _dirty_suffixes

        def row(key, target=0, finished=False):
            request = Mock(request_id=key, num_computed_tokens=target)
            request.is_finished.return_value = finished
            return request

        preempted = row('preempted')
        finished = row('finished', 200, finished=True)
        reader = row('reader', 256)
        retired = row('reused', 192, finished=True)
        replacement = row('reused')
        blocks = {'finished': [0, 7, 11, 14], 'reader': [8, 14, 15]}
        manager = NS(block_pool=NS(null_block=NS(block_id=0)),
                     get_block_ids=lambda key: (blocks[key],))
        scheduler = NS(running=[reader], kv_holding_waiting=[], kv_cache_manager=manager,
                       requests={r.request_id: r for r in (preempted, finished, reader, replacement)})
        snapshots = [(r, target, [NS(block_id=i) for i in ids])
                     for r, target, ids in ((preempted, 192, [0, 10, 11]),
                                            (retired, 192, [0, 10, 16]))]
        recover, dirty = _dirty_suffixes(scheduler, {0, 10}, snapshots)
        self.assertEqual(dirty, {10, 11, 14, 15, 16})
        self.assertEqual(recover, {preempted: (1, 192), reader: (1, 256)})
