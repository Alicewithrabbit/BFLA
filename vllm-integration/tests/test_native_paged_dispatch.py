import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import torch
from bfla_vllm import execution
from bfla_vllm.protocol import PrefillPlan


class DispatchTests(unittest.TestCase):
    def test_owned_fresh_alias_reads_native_pages_without_materialization(self):
        plan=PrefillPlan((0,4096),(4096,),'fresh_prefill',(4096,),(4096,),('r0',),((0,0,4096,4096),))
        impl=NS(kv_sharing_target_layer_name='owner')
        with patch.object(execution,'extend',return_value='paged') as run, patch.object(execution,'materialize_alias_kv') as gather:
            out=execution.forward(impl,None,None,None,None,NS(bfla_plan=plan),None,
                heads=None,group=None,settings=None,layer_owner='alias')
        self.assertEqual(out,'paged')
        run.assert_called_once();gather.assert_not_called()

    def test_fresh_uses_packed_inputs_or_the_alias_owners_page_order(self):
        plan = PrefillPlan((0, 3), (3,), 'fresh_prefill')
        table = torch.tensor([[2, 0]])
        meta = NS(bfla_plan=plan, block_table=table)
        cache = torch.arange(3 * 1 * 2 * 4).reshape(3, 1, 2, 4)
        q = torch.zeros(4, 1, 2)
        k, v = q + 1, q + 2
        output = torch.empty_like(q)
        for owner in (None, 'owner'):
            impl = NS(kv_sharing_target_layer_name=owner, head_size=2, scale=0.5)
            consume = Mock(captures=0)
            with self.subTest(owner=owner), patch.object(execution, '_reuse_reservation', return_value=consume), \
                    patch.object(execution, 'COUNTS', execution.Counter()):
                self.assertIs(execution.forward(impl, q, k, v, cache, meta, output,
                              heads=None, group=None, settings=None), output)
                query, key, value, target = consume.call_args.args
                self.assertEqual((len(query), target.data_ptr()), (3, output.data_ptr()))
                if owner is None:
                    self.assertEqual((key.data_ptr(), value.data_ptr()), (k.data_ptr(), v.data_ptr()))
                else:
                    packed = torch.cat([cache[2], cache[0, :, :1]], dim=1).transpose(0, 1)
                    torch.testing.assert_close(key, packed[..., :2])
                    torch.testing.assert_close(value, packed[..., 2:])
