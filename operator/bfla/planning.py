"""CPU planning of causal supports, request boundaries and bounded selector batches."""
from dataclasses import dataclass, replace
from functools import lru_cache
from math import gcd
from .config import BFLAConfig
from .capabilities import supports
from ._limits import QUERY_TILE_ROWS, KV_BLOCK_TOKENS, SELECTOR_WAVE_ROWS, SELECTOR_BATCH_ROWS


def tail_boundary(prompt_length, ratio, config=BFLAConfig()):
    if prompt_length <= config.short_full_threshold:
        return 0
    if config.tail_tiles == 0:
        return prompt_length
    tiles = (prompt_length * ratio + QUERY_TILE_ROWS - 1) // QUERY_TILE_ROWS
    return max(0, tiles - config.tail_tiles) * QUERY_TILE_ROWS // ratio


def mandatory_blocks(first, last, config=BFLAConfig()):
    b = config.pool_block_size
    end = last // b + 1
    local = min(first // b, max(0, (first // config.local_alignment - config.local_blocks + 1)
                               * config.local_alignment) // b)
    return tuple(sorted(set(range(min(end, (config.sink_tokens + b - 1) // b))) |
                        set(range(local, end))))


def mandatory_count(first, last, config=BFLAConfig()):
    """Size of the sink/local interval union, without materializing block IDs."""
    b = config.pool_block_size
    end = last // b + 1
    sink = min(end, (config.sink_tokens + b - 1) // b)
    local = min(first // b, max(0, (first // config.local_alignment - config.local_blocks + 1)
                               * config.local_alignment) // b)
    return sink + end - max(sink, local)


@dataclass(frozen=True)
class Wave:
    start: int
    rows: int
    sparse_rows: int
    sparse_tiles: int
    counts: tuple

    @property
    def tiles(self): return len(self.counts)


@dataclass(frozen=True)
class Group:
    request: int
    wave: int
    index: int
    first: int
    last: int
    count: int
    size: int
    consumer_tiles: tuple


@dataclass(frozen=True)
class Request:
    query_start: int
    key_length: int
    prefix: int
    pool_start: int
    waves: tuple

    @property
    def sparse_rows(self): return sum(w.sparse_rows for w in self.waves)


@dataclass(frozen=True)
class SelectorBatch:
    spans: tuple
    query_indices: tuple
    rows: tuple
    groups: tuple


@dataclass(frozen=True)
class PrefillMetadata:
    requests: tuple
    groups: tuple
    counts: tuple
    dense_tiles: tuple
    layout: dict
    batches: tuple


def _waves(n, prefix, prompt, ratio, dim, config):
    sparse = 0 if prefix + n <= config.short_full_threshold else min(
        n, max(0, tail_boundary(prompt, ratio, config) - prefix))
    return _waves_from_sparse(n, prefix, sparse, ratio, dim, config)


def _waves_from_sparse(n, prefix, sparse, ratio, dim, config):
    alignment = QUERY_TILE_ROWS // gcd(QUERY_TILE_ROWS, ratio)
    step = min(sparse or n, SELECTOR_WAVE_ROWS // alignment * alignment)
    parts = []
    if sparse:
        for start in range(0, sparse, step):
            rows = min(step, sparse - start)
            parts.append((start, rows, rows))
        if dim == 512 and sparse * ratio % QUERY_TILE_ROWS == 0:
            start, _, count = parts[-1]
            parts[-1] = (start, n - start, count)
        elif sparse < n:
            parts.append((sparse, n - sparse, 0))
    else:
        parts.append((0, n, 0))
    waves = []
    for start, rows, active in parts:
        active_tiles = (active * ratio + QUERY_TILE_ROWS - 1) // QUERY_TILE_ROWS
        counts = []
        for tile in range((rows * ratio + QUERY_TILE_ROWS - 1) // QUERY_TILE_ROWS):
            # Full-support suffix uses physical block64, independent of pooling.
            last = prefix + start + (min((tile + 1) * QUERY_TILE_ROWS, rows * ratio) - 1) // ratio
            counts.append(last // KV_BLOCK_TOKENS + 1)
        waves.append(Wave(start, rows, active, active_tiles, tuple(counts)))
    return waves


def _batch_layout(lengths, segments, ratio, kv_lengths, native_paged=False):
    pages = [(n + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS for n in kv_lengths]
    width = max(pages)
    cu_q, cu_tiles, used, table, offsets = [0], [0], [], [], []
    if native_paged:
        table = None
    page = max_rows = 0
    segment_requests = []
    for ri,(n, k, parts, npages) in enumerate(zip(lengths, kv_lengths, segments, pages)):
        offsets.append(page)
        end = k - n
        for rows in parts:
            end += rows
            segment_requests.append(ri)
            used.append(end)
            cu_q.append(cu_q[-1] + rows)
            cu_tiles.append(cu_tiles[-1] + (rows * ratio + QUERY_TILE_ROWS - 1) // QUERY_TILE_ROWS)
            if table is not None:
                table.append(list(range(page, page + npages)) + [-1] * (width - npages))
            max_rows = max(max_rows, rows)
        page += npages
    stride = (len(used) + 3) // 4 * 4
    return dict(cu_q=cu_q, cu_tiles=cu_tiles, used_k=used, table=table,
                request_page_offsets=offsets, total_pages=page, max_rows=max_rows, segment_requests=segment_requests,
                scheduler_stride=stride, scheduler_words=4 * stride + 1)


def make_plan(request_lengths, heads, config=BFLAConfig(), *, kv_lengths=None, prompt_lengths=None,
              native_paged=False):
    lengths = tuple(request_lengths)
    klens = lengths if kv_lengths is None else tuple(kv_lengths)
    prompts = klens if prompt_lengths is None else tuple(prompt_lengths)
    supports(heads, config, request_lengths=lengths, kv_lengths=klens, prompt_lengths=prompts,
             kv_layout='paged' if native_paged else 'ragged').require()
    sparse_rows = tuple(0 if k <= config.short_full_threshold else
        min(n, max(0, tail_boundary(p, heads.selection_ratio, config) - (k-n)))
        for n,k,p in zip(lengths,klens,prompts))
    # The prompt only affects the clipped suffix boundary. Reuse geometry
    # across compatible chunks without caching request state or selected IDs.
    if not native_paged or sum(lengths) > SELECTOR_BATCH_ROWS:
        return _assemble_plan(lengths, heads, config, klens, sparse_rows, native_paged)
    plan = _cached_plan(lengths, heads, config, klens, sparse_rows, native_paged)
    # Callers may rebind physical layouts. Keep mutable containers private;
    # all other records and their nested members are immutable.
    layout = {k: v[:] if isinstance(v, list) else v for k, v in plan.layout.items()}
    return replace(plan, layout=layout)


def _assemble_plan(lengths, heads, config, klens, sparse_rows, native_paged):
    ratio = heads.selection_ratio
    share, expand = config.query_tile_rows // QUERY_TILE_ROWS, config.pool_block_size // KV_BLOCK_TOKENS
    requests, groups, counts, dense, segments = [], [], [], [], []
    query_start = pool_start = tile_start = 0
    wave_groups = {}
    for ri, (n, kn, sparse) in enumerate(zip(lengths, klens, sparse_rows)):
        prefix = kn - n
        waves = _waves_from_sparse(n, prefix, sparse, ratio, heads.head_dim, config)
        for wi, wave in enumerate(waves):
            wc = list(wave.counts)
            for start in range(0, wave.sparse_tiles, share):
                stop = min(start + share, wave.sparse_tiles)
                first = prefix + wave.start + start * QUERY_TILE_ROWS // ratio
                last = prefix + wave.start + (min(stop * QUERY_TILE_ROWS, wave.sparse_rows * ratio) - 1) // ratio
                count, size = config.block_count(last + 1), last // config.pool_block_size + 1
                if mandatory_count(first, last, config) > count:
                    raise ValueError('mandatory blocks exceed selection quota')
                physical = count * expand - (size * expand - (last // KV_BLOCK_TOKENS + 1))
                for ti in range(start, stop): wc[ti] = physical
                group = Group(ri, wi, start // share, first, last, count, size,
                              tuple(range(tile_start + start, tile_start + stop)))
                groups.append(group)
                wave_groups.setdefault((ri, wi), []).append(group)
            counts.extend(wc)
            dense.extend(range(tile_start + wave.sparse_tiles, tile_start + wave.tiles))
            tile_start += wave.tiles
            waves[wi] = replace(wave, counts=tuple(wc))
        parts = tuple(w.rows for w in waves)
        if heads.consumer_ratio != ratio:
            # Replicated KV: each consumer segment retains one global support
            # tile. Capability checks restrict this to the tested divisible G4.
            tokens_per_tile = QUERY_TILE_ROWS // ratio
            parts = tuple(min(tokens_per_tile, w.rows - i) for w in waves
                          for i in range(0, w.rows, tokens_per_tile))
        segments.append(parts)
        requests.append(Request(query_start, kn, prefix, pool_start, tuple(waves)))
        query_start += n
        pool_start += (kn + config.pool_block_size - 1) // config.pool_block_size
    layout = _batch_layout(lengths, segments, heads.consumer_ratio, klens, native_paged)
    if len(counts) != layout['cu_tiles'][-1]:
        raise ValueError('selection and consumer tile maps differ')
    # Round-robin waves across requests, bounded to the existing 16K feature arena.
    ordered = [(ri, wi) for wi in range(max(len(r.waves) for r in requests))
               for ri, r in enumerate(requests) if wi < len(r.waves) and r.waves[wi].sparse_rows]
    batches, spans, indices, rows, selected_groups = [], [], [], [], []
    for ri, wi in ordered:
        r, w = requests[ri], requests[ri].waves[wi]
        if indices and len(indices) + w.sparse_rows > SELECTOR_BATCH_ROWS:
            batches.append(SelectorBatch(tuple(spans), tuple(indices), tuple(rows), tuple(selected_groups)))
            spans, indices, rows, selected_groups = [], [], [], []
        offset = len(indices)
        spans.append((r.query_start + w.start, w.sparse_rows, offset))
        indices.extend(range(r.query_start + w.start, r.query_start + w.start + w.sparse_rows))
        nb = (r.key_length + config.pool_block_size - 1) // config.pool_block_size
        for g in wave_groups[(ri, wi)]:
            rows.append((offset, r.pool_start, nb, r.prefix + w.start, w.sparse_rows, g.index))
            selected_groups.append(g)
    if indices:
        batches.append(SelectorBatch(tuple(spans), tuple(indices), tuple(rows), tuple(selected_groups)))
    return PrefillMetadata(tuple(requests), tuple(groups), tuple(counts), tuple(dense), layout, tuple(batches))


_cached_plan = lru_cache(maxsize=64)(_assemble_plan)
