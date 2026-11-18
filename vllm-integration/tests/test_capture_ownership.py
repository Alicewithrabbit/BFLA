"""CPU regression for explicit reservation retirement and stream ownership."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'bfla_vllm'


class CaptureOwnership(unittest.TestCase):
    def test_runtime_owns_nonpooled_stream_and_orders_reuse(self):
        source = (ROOT/'runtime.py').read_text()
        self.assertIn('CaptureLifetime', source)
        self.assertNotIn('torch.cuda.Stream()', source)
        self.assertIn('wait_for_last_use', source)
        self.assertIn('record_use', source)
        tree = ast.parse(source)
        self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == 'close'
                            for n in ast.walk(tree)))

    def test_shared_reservation_owner_closes_evicted_plans(self):
        source = (ROOT/'execution.py').read_text()
        self.assertIn('retired.close()',source)
        self.assertIn('execution.close()',source)
        from bfla_vllm import execution
        from bfla_vllm.qsa import consumer
        self.assertIs(consumer.RESERVATIONS,execution.RESERVATIONS)
        self.assertIs(consumer.POOLS,execution.POOLS)
        self.assertIs(consumer.release,execution.release)
        self.assertIs(consumer.consume_rows,execution.consume_prefills)


if __name__ == '__main__':
    unittest.main()
