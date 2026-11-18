"""Opt-in DI64/256 features and scaled BF16 pooled storage.

Q/K/V remain BF16 at native D. Key token features remain BF16 and query
features remain FP32 until the existing BF16 score dot. DI is independent
of the 128 packed query rows and the physical 64-token consumer tile.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def fused_features(X, Out, H: tl.constexpr, BH: tl.constexpr, D: tl.constexpr,
                   DI: tl.constexpr, RS: tl.constexpr, HS: tl.constexpr,
                   KEY: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, D)
    index = tl.arange(0, DI)
    if KEY:
        # Compact each normalized key head, then sum heads in order.
        compact = tl.full((DI,), 0., tl.float32)
        for head in tl.static_range(H):
            x = tl.load(X + row * RS + head * HS + d).to(tl.float32)
            norm = tl.rsqrt(tl.sum(x * x, 0) / D + 1.e-6)
            window_sum = tl.sum(tl.reshape(x, (DI, D // DI)), 1)
            compact += window_sum * norm / (H * tl.sqrt(float(D // DI)))
    else:
        h = tl.arange(0, BH)
        x = tl.load(X + row * RS + h[:, None] * HS + d[None, :],
                    h[:, None] < H, 0.).to(tl.float32)
        norm = tl.rsqrt(tl.sum(x * x, 1) / D + 1.e-6)
        mean = tl.sum(x * norm[:, None], 0) / H
        compact = tl.sum(tl.reshape(mean, (DI, D // DI)), 1) / tl.sqrt(float(D // DI))
    y = compact * tl.rsqrt(tl.sum(compact * compact, 0) / DI + 1.e-6)
    for bit in tl.static_range(0, 8):
        if (1 << bit) < DI:
            pair = tl.gather(y, index ^ (1 << bit), 0)
            y = tl.where((index & (1 << bit)) == 0, y + pair, pair - y)
    tl.store(Out + row * DI + index, y * (DI ** -.5))


class GeneralizedFeatures:
    def __init__(self, layout, index_dim):
        if layout.tp_size != 1 or layout.head_dim not in (256, 512):
            raise ValueError('this feature wrapper requires TP1 native D256/D512')
        if index_dim not in (64, 128, 256) or layout.head_dim % index_dim:
            raise ValueError('index dimension must partition the native dimension')
        self.layout, self.index_dim = layout, index_dim

    def _run(self, x, out, key):
        heads = self.layout.local_hkv if key else self.layout.local_hq
        expected_dtype = torch.bfloat16 if key else torch.float32
        if (x.ndim != 3 or x.shape[1:] != (heads, self.layout.head_dim) or
                not x.is_cuda or x.dtype != torch.bfloat16 or x.stride(-1) != 1 or
                out.shape != (x.shape[0], self.index_dim) or out.dtype != expected_dtype or
                out.device != x.device or not out.is_contiguous()):
            raise ValueError('feature geometry, dtype or output ownership changed')
        fused_features[(x.shape[0],)](x, out, heads, tr.next_power_of_2(heads),
            self.layout.head_dim, self.index_dim, x.stride(0), x.stride(1), key,
            num_warps=1, enable_fp_fusion=False)
        return out

    def key(self, k, out):
        return self._run(k, out[:k.shape[0]], True)

    def query_batch(self, q, out, spans):
        for source, rows, destination in spans:
            self._run(q[source:source+rows], out[destination:destination+rows], False)


@tr.jit(do_not_specialize=['N'])
def pool_index(Features, Pool, Scales, N, DI: tl.constexpr,
               B: tl.constexpr = 64, BF16: tl.constexpr = False):
    b = tl.program_id(0)
    token = b * B + tl.arange(0, B)
    d = tl.arange(0, DI)
    x = tl.load(Features + token[:, None] * DI + d[None, :],
                token[:, None] < N, 0.).to(tl.float32)
    mean = tl.sum(x, 0) / tl.minimum(B, N - b * B)
    scale = tl.maximum(tl.max(tl.abs(mean), 0), 1.e-6) / 448.
    # FP8 and BF16 storage share the same FP32 scale. BF16 is cast directly
    # from the scaled mean.
    if BF16:
        stored = (mean / scale).to(tl.bfloat16)
    else:
        stored = (mean / scale).to(tl.float8e4nv)
    tl.store(Pool + b * DI + d, stored)
    tl.store(Scales + b, scale)


@tr.jit(do_not_specialize=['STRIDE'])
def score_index(Q, Pool, Scales, Scores, Rows, G: tl.constexpr, STRIDE,
                DI: tl.constexpr, M: tl.constexpr = 128, B: tl.constexpr = 64,
                PRUNED: tl.constexpr = False, BN: tl.constexpr = 32):
    output = tl.program_id(0)
    qbase = tl.load(Rows + output * 6)
    poolbase = tl.load(Rows + output * 6 + 1)
    nb = tl.load(Rows + output * 6 + 2)
    start = tl.load(Rows + output * 6 + 3)
    rows = tl.load(Rows + output * 6 + 4)
    tile = tl.load(Rows + output * 6 + 5)
    first = start + tile * M // G
    last = start + (tl.minimum((tile + 1) * M, rows * G) - 1) // G
    bs = tl.program_id(1) * BN
    b = bs + tl.arange(0, BN)
    local = tl.minimum(first // B, tl.maximum(0, (first // 128 - 3) * 128) // B)
    sink = tl.constexpr((256 + B - 1) // B)
    if bs >= nb:
        energy = tl.full((BN,), -float('inf'), tl.float32)
    elif PRUNED and bs > last // B:
        energy = tl.full((BN,), -float('inf'), tl.float32)
    elif PRUNED and ((bs + BN <= sink) | (bs >= local)):
        energy = tl.where(b <= last // B, float('inf'), -float('inf'))
    else:
        d = tl.arange(0, DI)
        k = tl.load(Pool + (poolbase + b[None, :]) * DI + d[:, None],
                    b[None, :] < nb, 0.).to(tl.bfloat16)
        scale = tl.load(Scales + poolbase + b, b < nb, 0.)
        if M == 128:
            # Preserve the original score_rows order for the default tile.
            packed = tile * 128 + tl.arange(0, 128)
            row = packed // G
            valid = packed < rows * G
            q = tl.load(Q + (qbase + row[:, None]) * DI + d[None, :],
                        valid[:, None], 0.).to(tl.bfloat16)
            s = tl.maximum(tl.dot(q, k), 0) * (DI ** -.5)
            s *= scale[None, :]
            s = tl.where(valid[:, None], s, -float('inf'))
            peak = tl.max(s, 0)
            energy = peak + tl.log(tl.sum(tl.exp(s - peak[None, :]), 0))
        else:
            peak = tl.full((BN,), -float('inf'), tl.float32)
            mass = tl.full((BN,), 0., tl.float32)
            for part in range(M // 128):
                packed = tile * M + part * 128 + tl.arange(0, 128)
                row = packed // G
                valid = packed < rows * G
                # A short final sharing group can have wholly empty parts.
                # Skipping them prevents (-inf)-(-inf) from producing NaN.
                if tl.sum(valid.to(tl.int32), 0) > 0:
                    q = tl.load(Q + (qbase + row[:, None]) * DI + d[None, :],
                                valid[:, None], 0.).to(tl.bfloat16)
                    s = tl.maximum(tl.dot(q, k), 0) * (DI ** -.5) * scale[None, :]
                    s = tl.where(valid[:, None], s, -float('inf'))
                    next_peak = tl.maximum(peak, tl.max(s, 0))
                    mass = mass * tl.exp(peak - next_peak) + tl.sum(tl.exp(s - next_peak[None, :]), 0)
                    peak = next_peak
            energy = peak + tl.log(mass)
        energy = tl.where((b < sink) | (b >= local), float('inf'), energy)
        energy = tl.where(b <= last // B, energy, -float('inf'))
    tl.store(Scores + output.to(tl.int64) * STRIDE + b, energy, b < STRIDE)

