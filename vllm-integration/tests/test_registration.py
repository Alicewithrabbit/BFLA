"""No runtime import side effects, and explicit native backend registration."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch


class Registration(unittest.TestCase):
    def test_import_and_disabled_plugin_need_neither_torch_nor_vllm(self):
        env = dict(os.environ, BFLA_QSA_ENABLED='0')
        env.pop('BFLA_NATIVE_BACKEND', None)
        code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'triton', 'vllm'):
        raise AssertionError('unexpected runtime dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import bfla_vllm
from bfla_vllm import protocol, settings
from bfla_vllm.qsa import adapter, decode_selection
bfla_vllm.register()
"""
        subprocess.run([sys.executable, '-c', code], check=True, env=env)

    def test_target_uses_framework_version_and_backend_without_model_dispatch(self):
        from bfla_vllm import backend_target
        self.assertEqual(backend_target('flash_attn', '0.31.0rc1'),
                         'bfla_vllm.v031.FlashPrefillBackend')
        self.assertEqual(backend_target('triton', '0.31.0'),
                         'bfla_vllm.v031.TritonPrefillBackend')
        for backend, version in [('other', '0.31.0'), ('triton', '0.32.0')]:
            with self.assertRaises(ValueError):
                backend_target(backend, version)


    def test_old_framework_is_rejected_for_both_integrations(self):
        from bfla_vllm import register
        for settings in ({'BFLA_NATIVE_BACKEND': 'triton'}, {'BFLA_QSA_ENABLED': '1'}):
            with self.subTest(settings=settings), patch.dict(os.environ, settings, clear=True), \
                    patch('importlib.metadata.version', return_value='0.19.1'), \
                    self.assertRaisesRegex(ValueError, r'requires vLLM 0\.31\.x, found 0\.19\.1\.$'):
                register()


if __name__ == '__main__':
    unittest.main()
