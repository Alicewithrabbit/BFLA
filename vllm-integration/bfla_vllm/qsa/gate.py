# Derived from vLLM, Apache-2.0, Copyright vLLM contributors.
import triton as tr
import triton.language as tl

@tr.jit(do_not_specialize=['N'])
def apply_gate(X, Gate, Out, N, BLOCK: tl.constexpr = 1024):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    gate = tl.load(Gate + i, i < N, 0).to(tl.float32)
    tl.store(Out + i, x * tl.sigmoid(gate), i < N)

