"""Host dispatch selects exactly the key-feature implementation it will use."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bfla import HeadLayout, build
from bfla.features import FusedFeatures


class FeatureDispatchTests(unittest.TestCase):
    def test_only_the_selected_key_extension_is_loaded(self):
        query256, query512 = object(), object()
        modules = {
            'bfla._kernels.query_d256': SimpleNamespace(query_features=query256),
            'bfla._kernels.query_d512': SimpleNamespace(query_features=query512),
        }
        for heads, uses_warp, query in ((HeadLayout.create(24,4,256),True,query256),
                                        (HeadLayout.create(48,8,256),False,query256),
                                        (HeadLayout.create(8,2,512),False,query512)):
            with self.subTest(heads=heads), patch.dict('sys.modules',modules), \
                    patch.object(build,'key_feature') as warp, \
                    patch.object(build,'feature_extension') as generic:
                features=FusedFeatures(heads)
                self.assertIs(features.query,query)
                if uses_warp:
                    warp.assert_called_once_with()
                    generic.assert_not_called()
                    self.assertIs(features.warp,warp.return_value)
                else:
                    generic.assert_called_once_with()
                    warp.assert_not_called()
                    self.assertIs(features.extension,generic.return_value)


if __name__ == '__main__': unittest.main()
