"""Run package tests from the checkout without starting services or inference jobs."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
env = dict(os.environ)
env['PYTHONPATH'] = os.pathsep.join(str(root/p) for p in ('operator','vllm-integration'))
for tests in ('operator/tests','vllm-integration/tests'):
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(root/tests),'-v'],env=env,check=True)
subprocess.run([sys.executable,str(root/'tools/audit_forward_headers.py')],check=True)
