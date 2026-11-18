import ast
from pathlib import Path
import subprocess
import sys
import unittest


class ArchitectureTests(unittest.TestCase):
    def test_cpu_api_has_no_optional_dependency_import(self):
        root = Path(__file__).resolve().parents[1]
        code = (f'import sys; sys.path.insert(0,{str(root)!r}); '
                'import bfla; from bfla import IndexPoolState; '
                'assert not any(n in sys.modules for n in ("torch","triton","vllm"))')
        subprocess.run([sys.executable,'-I','-S','-c',code],check=True)

    def test_operator_does_not_import_framework_or_modify_paths(self):
        root = Path(__file__).resolve().parents[1]/'bfla'
        files = list(root.glob('*.py'))+list((root/'_kernels').glob('*.py'))
        for file in files:
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                imports = [a.name for a in node.names] if isinstance(node,ast.Import) else (
                    [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                self.assertFalse(any(n.split('.')[0] in ('vllm','transformers') for n in imports),file)
            text = file.read_text().lower()
            self.assertNotIn('sys.path',text,str(file))
            self.assertNotIn('os.environ',text,str(file))


if __name__ == '__main__': unittest.main()
