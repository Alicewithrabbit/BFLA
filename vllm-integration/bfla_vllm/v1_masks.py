"""Preserve vLLM 0.31 attention masks and request phases across V1 microbatches."""
from functools import wraps


def _image_ranges(runner, position):
    # V1 builds these inclusive ranges inline in _build_attention_metadata.
    config = runner.model_config.hf_text_config
    modulus = getattr(config, 'mm_prefix_span_leading_pad_modulus', 0)
    if modulus:
        pad = modulus - 1 - position.offset % modulus
        ranges = [(position.offset + pad, position.offset + position.length - 1)]
    else:
        ranges = position.extract_embeds_range()
    window = getattr(config, 'sliding_window', None)
    clamp = (getattr(runner.model, 'mm_prefix_clamp_sliding_window', False)
             or getattr(config, 'mm_prefix_clamp_sliding_window', False))
    for lo, hi in ranges:
        if clamp or window is None or hi - lo + 1 <= window:
            yield lo, hi


def _cut_strides(runner, total):
    # V1 cuts at j * floor(padded_tokens / n). DP can adopt a peer's
    # SP/graph-padded length, provided the local last slice stays nonempty.
    n = runner.parallel_config.num_ubatches
    if n < 2:
        return n, 0, 0
    cap = max(runner._pad_for_sequence_parallelism(runner.max_num_tokens),
              runner.compilation_config.max_cudagraph_capture_size or 0)
    return n, total // n, min(cap // n, (total - 1) // (n - 1))


def _cut_inside(start, end, strides):
    n, shortest, longest = strides
    for j in range(1, n):
        if max(shortest, start // j + 1) <= min(longest, (end - 1) // j):
            return True
    return False


def _may_split_image(runner, counts):
    if not runner.is_mm_prefix_lm:
        return False
    batch = runner.input_batch
    packed_stop = 0
    strides = None
    for row, count in enumerate(counts):
        count = int(count)
        packed_stop += count
        if count <= 1:
            continue
        start = int(batch.num_computed_tokens_cpu[row])
        stop = start + count
        for feature in runner.requests[batch.req_ids[row]].mm_features:
            if feature.modality == 'audio':
                continue
            pos = feature.mm_position
            if min(stop, pos.offset + pos.length) - max(start, pos.offset) < 2:
                continue
            for lo, hi in _image_ranges(runner, pos):
                left, right = max(start, lo), min(stop, hi + 1)
                if right - left < 2:
                    continue
                if strides is None:
                    strides = _cut_strides(runner, int(sum(counts)))
                row_offset = packed_stop - stop
                if _cut_inside(row_offset + left, row_offset + right, strides):
                    return True
    return False


def _may_split_aligned_state(runner, counts):
    # V1 preprocesses aligned Mamba state once, into the full step's last
    # block. A straddling request has no native handoff between microbatches.
    if runner.cache_config.mamba_cache_mode != 'align':
        return False
    total = int(counts.sum())
    if total == len(counts):
        return False
    strides = _cut_strides(runner, total)
    end = 0
    for count in counts:
        start = end
        end += int(count)
        if count > 1 and _cut_inside(start, end, strides):
            return True
    return False


def _masked_split(original):
    @wraps(original)
    def split(ubatch_slices, common_attn_metadata):
        cm = common_attn_metadata
        parts = original(ubatch_slices, cm)
        ranges = cm.mm_req_doc_ranges
        if (cm.is_prefilling is None and cm.causal is True
                and ranges is None and cm.rswa_prefix_lens is None):
            return parts
        for cut, part in zip(ubatch_slices, parts):
            rows = cut.request_slice
            # GDN uses this flag to reset recycled state on a fresh one-token prefill.
            if cm.is_prefilling is not None:
                part.is_prefilling = cm.is_prefilling[rows]
            last = rows.stop - 1
            if cm.causal is not True or ranges and ranges.get(last):
                remaining = int(cm.query_start_loc_cpu[last + 1]) - cut.token_slice.stop
                if remaining > 0:
                    if cm.causal is not True:
                        raise ValueError('cannot split a request with a noncausal or dynamic mask')
                    boundary = int(cm.seq_lens_cpu_upper_bound[last]) - remaining
                    if any(a < boundary <= b for a, b in ranges[last]):
                        raise ValueError('microbatch boundary crosses a bidirectional range')
            part.causal = cm.causal if isinstance(cm.causal, bool) else cm.causal[rows]
            if ranges is not None:
                part.mm_req_doc_ranges = {i - rows.start: ranges[i]
                                          for i in range(rows.start, rows.stop) if i in ranges}
            if cm.rswa_prefix_lens is not None:
                part.rswa_prefix_lens = cm.rswa_prefix_lens[rows]
        return parts
    split._bfla_mask_slices = True
    return split


def install(runner_class):
    from vllm.v1.worker import gpu_model_runner
    split = gpu_model_runner.split_attn_metadata
    if not getattr(split, '_bfla_mask_slices', False):
        gpu_model_runner.split_attn_metadata = _masked_split(split)
    allow = runner_class._allow_microbatching

    @wraps(allow)
    def allow_microbatching(self, num_reqs, num_scheduled_tokens_np):
        if self.parallel_config.use_ubatching:
            counts = num_scheduled_tokens_np[:num_reqs]
            if _may_split_aligned_state(self, counts) or _may_split_image(self, counts):
                return False
        return allow(self, num_reqs, num_scheduled_tokens_np)

    runner_class._allow_microbatching = allow_microbatching
