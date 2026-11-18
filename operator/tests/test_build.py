"""Check selector dispatch without CUDA or extension compilation."""
import unittest
from unittest.mock import patch

from bfla import build


class SelectorDispatchTests(unittest.TestCase):
    def test_dispatch_covers_both_selected_and_candidate_capacity(self):
        cases = (
            (1, 1, 512), (512, 8192, 512),
            (513, 2048, 1024), (1024, 2048, 1024),
            (513, 2049, 2048), (1024, 8192, 2048),
            (1025, 8192, 2048), (2048, 8192, 2048),
        )
        with patch.object(build, '_selector', side_effect=lambda bound: bound) as load:
            for capacity, candidates, expected in cases:
                with self.subTest(capacity=capacity, candidates=candidates):
                    self.assertEqual(build.selector(capacity, candidates=candidates), expected)
            self.assertEqual(load.call_count, len(cases))

    def test_unsupported_geometry_is_rejected_before_loading(self):
        with patch.object(build, '_selector') as load:
            for capacity, candidates in ((0, 128), (2049, 8192), (512, 8193),
                                         (512, 0), (True, 1024), (512, 2.5)):
                with self.subTest(capacity=capacity, candidates=candidates):
                    with self.assertRaises(ValueError):
                        build.selector(capacity, candidates=candidates)
            load.assert_not_called()

    def test_capacity_only_call_retains_existing_dispatch(self):
        with patch.object(build, '_selector', side_effect=lambda bound: bound):
            self.assertEqual(build.selector(), 512)
            self.assertEqual(build.selector(1024), 1024)
            self.assertEqual(build.selector(2048), 2048)


if __name__ == '__main__':
    unittest.main()
