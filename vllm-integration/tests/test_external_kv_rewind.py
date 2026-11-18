"""A running request can retry external KV under the same request and layer ID."""
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from bfla import BFLAConfig, HeadLayout, IndexPoolState
from bfla_vllm import execution
from bfla_vllm.protocol import classify_prefill
from bfla_vllm.settings import Settings


CONFIG = BFLAConfig(index_dtype='bf16')
HEADS = HeadLayout.create(6, 1, 256)


def pool_blocks(features, pool, scales, n, config):
    block = config.pool_block_size
    for i in range(len(pool)):
        mean = features[i*block:min(n, (i+1)*block)].float().mean(0)
        scales[i] = mean.abs().max()
        pool[i].copy_(mean / scales[i])


class CpuFeatures:
    layout = HEADS

    def key(self, key, out):
        out[:len(key)].copy_(key[:, 0, :CONFIG.index_dim])


def build_pool(state, keys, length):
    blocks = (length + CONFIG.pool_block_size - 1) // CONFIG.pool_block_size
    pool = torch.empty((blocks, CONFIG.index_dim), dtype=torch.bfloat16)
    scales = torch.empty(blocks, dtype=torch.float32)
    scratch = torch.empty((min(length, 1024), CONFIG.index_dim), dtype=torch.bfloat16)
    refreshed = state.populate_from_reader(
        CpuFeatures(), length, lambda start, end: keys[start:end], scratch, pool, scales,
        key_shape=(1, 256), key_dtype=keys.dtype, window=1024)
    return pool, scales, refreshed


class BindingRecorded(Exception):
    pass


class ExternalKvRewind(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(execution, 'POOLS', {}))
        stack.enter_context(patch.object(execution, 'IndexPoolState',
                                        side_effect=lambda config: IndexPoolState(config, _pooler=pool_blocks)))

    def bound_states(self, layer_owner, specs):
        keys, starts, visible = [], [0], []
        for request_id, history, length in specs:
            keys.append(request_id)
            starts.append(starts[-1] + length - history)
            visible.append(length)
        route = classify_prefill(starts, visible, True, chunking_enabled=True,
                                 prompt_lengths=[32768] * len(keys), request_keys=keys,
                                 is_prefilling=[True] * len(keys))
        self.assertEqual(route.reason, 'extend_prefill')
        # Stop at the GPU execution boundary. The adapter resolved the real owner states.
        width = (max(visible) + 63) // 64
        table = torch.empty((len(keys), width), dtype=torch.int32, device='meta')
        cache = torch.empty((len(keys)*width, 1, 64, 512), dtype=torch.bfloat16, device='meta')
        q = torch.empty((starts[-1], 6, 256), dtype=torch.bfloat16, device='meta')
        reservation = Mock()
        reservation.bind_paged_kv.side_effect = BindingRecorded
        impl = SimpleNamespace(kv_sharing_target_layer_name=None, scale=256**-.5)
        with patch.object(execution, '_reuse_reservation', return_value=reservation), self.assertRaises(BindingRecorded):
            execution.forward(impl, q, None, None, cache, SimpleNamespace(bfla_plan=route, block_table=table),
                              q, heads=HEADS, group=None, settings=Settings(CONFIG), layer_owner=layer_owner)
        return dict(zip(keys, reservation.bind_paged_kv.call_args.args[-1]))

    def check_retries(self, first, second):
        initial = self.bound_states('global.1', [(key, 4096, 4352) for key in ('first', 'shared', 'append')])
        other = self.bound_states('global.2', [('first', 4096, 4352)])['first']
        keys = torch.ones((4608, 1, 256), dtype=torch.bfloat16)
        for state in [*initial.values(), other]:
            build_pool(state, keys, 4352)

        repaired = keys.clone()
        # Both requests share a repaired external page. The second scheduler row
        # may resume AFTER this page because the first row recomputes it for both.
        repaired[4032:4096, :, :64] = 5
        retries = [('shared', second, 4480), ('append', 4352, 4416), ('first', first, 4352)]
        states = self.bound_states('global.1', retries)
        self.assertEqual(other.filled, 4352)
        for key, _, length in retries:
            source = keys if key == 'append' else repaired
            pool, scales, refreshed = build_pool(states[key], source, length)
            expected_pool, expected_scales, _ = build_pool(IndexPoolState(CONFIG, _pooler=pool_blocks), source, length)
            with self.subTest(request=key):
                torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)
                torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
                if key == 'append':
                    self.assertEqual(refreshed, 64)
        _, other_scales, refreshed = build_pool(other, keys, 4352)
        self.assertEqual((other_scales[63].item(), refreshed), (1, 0))

    def test_same_request_retry_rebuilds_only_rewound_layer_owners(self):
        self.check_retries(first=4032, second=4096)

    def test_forward_gap_and_normal_append_reuse_completed_blocks(self):
        keys = torch.ones((4480, 1, 256), dtype=torch.bfloat16)
        for history, length, expected in [(4096, 4161, 4161), (4352, 4416, 256), (4416, 4480, 64)]:
            state = self.bound_states('global.1', [('apc', history, length)])['apc']
            _, _, refreshed = build_pool(state, keys, length)
            self.assertEqual(refreshed, expected)
