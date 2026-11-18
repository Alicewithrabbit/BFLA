"""Invocation-scoped CPU request ownership for ordinary vLLM 0.31 backends."""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, signature


@dataclass(frozen=True)
class RequestContext:
    keys: tuple
    prompt_lengths: tuple
    # V2 hybrid model states turn single prompt tokens into decode rows for
    # their recurrent kernels. Override those flags only for ordinary attention.
    single_token_phases: tuple = ()

    def prefill_flags(self, flags):
        if self.single_token_phases and len(flags) == len(self.keys):
            flags = list(flags)
            for row, genuine_prompt_token in self.single_token_phases:
                flags[row] = genuine_prompt_token
        return flags


current_requests = ContextVar('bfla_requests', default=None)
sparse_prefill_disabled = ContextVar('bfla_sparse_prefill_disabled', default=False)


def install_v1(runner_class):
    if getattr(runner_class._build_attention_metadata, '_bfla_request_context', False):
        return
    original = runner_class._build_attention_metadata
    if hasattr(runner_class, '_determine_batch_execution_and_padding'):
        install_v1_graphs(runner_class)
    if hasattr(runner_class, '_allow_microbatching'):
        from . import v1_masks
        v1_masks.install(runner_class)

    positional = [p.name for p in signature(original).parameters.values()
                  if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)]
    # Only extra positional arguments reach *args after self and the three required fields.
    extra_index = {name: i for i, name in enumerate(positional[4:])}

    def argument(name, args, kwargs, default):
        index = extra_index.get(name)
        return args[index] if index is not None and index < len(args) else kwargs.get(name, default)

    @wraps(original)
    def build(self, num_tokens, num_reqs, max_query_len, *args, **kwargs):
        capture = argument('for_cudagraph_capture', args, kwargs, False)
        sliced = argument('ubatch_slices', args, kwargs, None) is not None
        context = None
        if not capture:
            from .execution import retain_requests
            retain_requests(self.requests)
            if not sliced:
                keys = tuple(self.input_batch.req_ids[:num_reqs])
                prompts = tuple(self.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs].tolist())
                context = RequestContext(keys, prompts)
        disabled = sparse_prefill_disabled.set(sliced)
        token = current_requests.set(context)
        try:
            return original(self, num_tokens, num_reqs, max_query_len, *args, **kwargs)
        finally:
            current_requests.reset(token)
            sparse_prefill_disabled.reset(disabled)

    build._bfla_request_context = True
    runner_class._build_attention_metadata = build
    if hasattr(runner_class, 'initialize_metadata_builders'):
        initialize = runner_class.initialize_metadata_builders

        @wraps(initialize)
        def initialize_builders(self, *args, **kwargs):
            result = initialize(self, *args, **kwargs)
            from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
            self._bfla_has_gdn = False
            # V1 caches by (KV spec, builder type), omitting the microbatch.
            # Only builder 0 can rebind pages without crossing slice boundaries.
            for groups in self.attn_groups:
                for group in groups:
                    if isinstance(group.metadata_builders[0], GDNAttentionMetadataBuilder):
                        self._bfla_has_gdn = True
                    for builder in group.metadata_builders[1:]:
                        if builder.supports_update_block_table:
                            builder.supports_update_block_table = False
            return result

        runner_class.initialize_metadata_builders = initialize_builders

    _install_pipeline_init(runner_class, v2=False)

    if hasattr(runner_class, 'execute_model'):
        execute = runner_class.execute_model

        @wraps(execute)
        def execute_model(self, scheduler_output, *args, **kwargs):
            result = execute(self, scheduler_output, *args, **kwargs)
            from .execution import POOLS, release_requests
            if POOLS and scheduler_output.num_scheduled_tokens:
                # Native microbatches and graph replays bypass per-layer retirement.
                # V1 keeps this step's starting positions in the input batch.
                batch = self.input_batch
                drafts = scheduler_output.scheduled_spec_decode_tokens
                completed = set()
                for i, key in enumerate(batch.req_ids):
                    prefix = int(batch.num_computed_tokens_cpu[i])
                    prompt = int(batch.num_prompt_tokens[i])
                    if prefix < prompt:
                        valid = scheduler_output.num_scheduled_tokens.get(key, 0) - len(drafts.get(key, ()))
                        if prefix + valid >= prompt:
                            completed.add(key)
                release_requests(completed)
            return result

        runner_class.execute_model = execute_model

    if hasattr(runner_class, '_update_states'):
        update = runner_class._update_states

        @wraps(update)
        def update_states(self, scheduler_output):
            from .execution import release_requests
            invalid = (set(getattr(scheduler_output, 'bfla_repaired_req_ids', ()))
                | set(scheduler_output.finished_req_ids)
                | set(scheduler_output.preempted_req_ids or ())
                | set(scheduler_output.scheduled_cached_reqs.resumed_req_ids)
                | {r.req_id for r in scheduler_output.scheduled_new_reqs})
            release_requests(invalid)
            rewound = getattr(scheduler_output, 'bfla_rewound_req_ids', ())
            if not rewound or not getattr(self, 'use_async_scheduling', False):
                return update(self, scheduler_output)

            batch = self.input_batch
            for key in rewound:
                # Old draft lengths enqueue both a GPU position correction and
                # a deferred CPU correction. Old row indices also feed sampled
                # and draft IDs into the next input. Replay uses the cached IDs.
                self.requests[key].prev_num_draft_len = 0
                if batch.prev_req_id_to_index is not None:
                    batch.prev_req_id_to_index.pop(key, None)
            result = update(self, scheduler_output)
            if (not batch.prev_req_id_to_index and self.num_accepted_tokens_event is not None
                    and self.cache_config.mamba_cache_mode == 'align'):
                # With no old rows the native align path copies counts by
                # current index. Wait for the old D2H before resetting these.
                self.num_accepted_tokens_event.synchronize()
                for key in rewound:
                    batch.num_accepted_tokens_cpu[batch.req_id_to_index[key]] = 1
            return result

        runner_class._update_states = update_states


def install_v2(state_class):
    original = state_class.prepare_attn
    if getattr(original, '_bfla_request_context', False):
        return

    @wraps(original)
    def prepare(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                attn_groups, kv_cache_config, for_capture=False, *args, **kwargs):
        context = None
        if not for_capture and not sparse_prefill_disabled.get():
            n = input_batch.num_reqs
            keys = tuple(input_batch.req_ids[:n])
            owned_prompts = getattr(self, '_bfla_prompt_lengths', {})
            if all(key in owned_prompts for key in keys):
                phases = ()
                native_decode_rows = getattr(input_batch, 'prefill_runs_as_decode_np', None)
                if native_decode_rows is not None:
                    drafts = input_batch.num_draft_tokens_per_req
                    phases = tuple((i, bool(input_batch.num_scheduled_tokens[i] == 1
                                            and (drafts is None or drafts[i] == 0)))
                                   for i, remapped in enumerate(native_decode_rows[:n]) if remapped)
                context = RequestContext(keys, tuple(owned_prompts[key] for key in keys), phases)
        token = current_requests.set(context)
        try:
            return original(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                            attn_groups, kv_cache_config, for_capture, *args, **kwargs)
        finally:
            current_requests.reset(token)

    prepare._bfla_request_context = True
    state_class.prepare_attn = prepare
    add = state_class.add_request

    @wraps(add)
    def add_request(self, req_index, new_req_data):
        result = add(self, req_index, new_req_data)
        if not hasattr(self, '_bfla_prompt_lengths'):
            self._bfla_prompt_lengths = {}
        self._bfla_prompt_lengths[new_req_data.req_id] = int(new_req_data.prompt_len)
        # Re-admission after preemption may replace stored K even at the same
        # visible length; invalidate compact state on every admission.
        _invalidate_v2_request(self, new_req_data.req_id)
        return result

    state_class.add_request = add_request
    remove = state_class.remove_request

    @wraps(remove)
    def remove_request(self, request_id):
        result = remove(self, request_id)
        _invalidate_v2_request(self, request_id)
        getattr(self, '_bfla_prompt_lengths', {}).pop(request_id, None)
        return result

    state_class.remove_request = remove_request


def _invalidate_v2_request(state, request_id):
    pending = getattr(state, '_bfla_retiring', None)
    if pending is None:
        from .execution import release_request
        release_request(request_id)
    else:
        pending.add(request_id)


def _local_graph_ratios(runner):
    from bfla import supports
    from .backend import PrefillMixin
    from .protocol import attention_contract_reason, resolve_head_layout

    ratios = set()
    for layer in runner.compilation_config.static_forward_context.values():
        impl = getattr(layer, 'impl', None)
        if not isinstance(impl, PrefillMixin) or attention_contract_reason(impl, None):
            continue
        try:
            heads = resolve_head_layout(impl, layer, runner.model_config,
                tp_size=runner.parallel_config.tensor_parallel_size)
        except ValueError:
            continue
        if supports(heads, request_lengths=(1,)).supported:
            ratios.add(heads.selection_ratio)
    return tuple(sorted(ratios))


def _tp_can_select(runner):
    if runner.parallel_config.tensor_parallel_size == 1:
        return True
    from vllm.distributed import get_tp_group
    from .collectives import has_feature_reducer
    return has_feature_reducer(get_tp_group())


def _install_pipeline_init(runner_class, *, v2):
    initialize = getattr(runner_class, 'initialize_kv_cache', None)
    if initialize is None:
        return

    @wraps(initialize)
    def initialize_kv_cache(self, kv_cache_config, is_profiling=False, *args, **kwargs):
        result = initialize(self, kv_cache_config, is_profiling, *args, **kwargs)
        # The optional profiling call may run on only some stages. The final
        # initialization runs on all workers, after native backend resolution.
        if not is_profiling and self.parallel_config.pipeline_parallel_size > 1:
            _share_pipeline_graph_requirements(self, v2=v2)
        return result

    runner_class.initialize_kv_cache = initialize_kv_cache


def _share_pipeline_graph_requirements(runner, *, v2=False):
    import torch.distributed as dist
    from vllm.config import CUDAGraphMode
    from vllm.distributed import get_pp_group
    from . import pipeline

    group = get_pp_group()
    ratios = _local_graph_ratios(runner)
    if ratios and not _tp_can_select(runner):
        ratios = ()
    # Each PP stage owns different layers. The intermediate tensor's padded
    # length must agree, including on stages with no attention layers.
    compilation = runner.compilation_config
    mode = runner.cudagraph_manager.cudagraph_mode if v2 else compilation.cudagraph_mode
    graph = (mode, tuple(compilation.cudagraph_capture_sizes or ()))
    ordering = None if v2 else getattr(runner, 'reorder_batch_threshold', None)
    cascades = () if v2 else pipeline.cascade_policies(runner)
    local = (getattr(runner, '_bfla_has_gdn', False), ratios, graph, ordering, cascades)
    stages = [None] * group.world_size
    dist.all_gather_object(stages, local, group=group.cpu_group)
    stage_gdn, stage_ratios, stage_graphs, thresholds, policies = zip(*stages)
    runner._bfla_has_gdn = any(stage_gdn)
    runner._bfla_pipeline_ratios = tuple(sorted({r for ratios in stage_ratios for r in ratios}))
    # Speculative capture sizing can also differ between resolved backends.
    width = runner.decode_query_len if v2 else runner.uniform_decode_query_len
    decode_limit = pipeline.common_decode_limit(stage_graphs, runner, width)
    runner._bfla_pipeline_decode_limit = decode_limit
    mixed_graphs = {(stage_mode.mixed_mode(), sizes) for stage_mode, sizes in stage_graphs}
    runner._bfla_pipeline_force_eager = len(mixed_graphs) != 1 or mode.mixed_mode() == CUDAGraphMode.FULL
    if not v2:
        pipeline.align_v1(runner, thresholds, policies)
    elif runner._bfla_pipeline_force_eager or decode_limit is not None:
        pipeline.align_v2_graphs(runner.cudagraph_manager, runner._bfla_pipeline_force_eager, decode_limit)


def _may_select_blocks(runner, prefix, prompt, config):
    from bfla.planning import tail_boundary

    visible = prefix + 1
    if (visible > prompt or visible <= config.short_full_threshold
            or config.block_count(visible) * config.pool_block_size >= visible):
        return False
    ratios = getattr(runner, '_bfla_pipeline_ratios', None)
    if ratios is None:
        if not hasattr(runner, '_bfla_graph_ratios'):
            runner._bfla_graph_ratios = _local_graph_ratios(runner)
        ratios = runner._bfla_graph_ratios
    if not any(prefix < tail_boundary(prompt, ratio, config) for ratio in ratios):
        return False
    return hasattr(runner, '_bfla_pipeline_ratios') or _tp_can_select(runner)


def install_v1_graphs(runner_class):
    determine = runner_class._determine_batch_execution_and_padding
    if getattr(determine, '_bfla_prompt_dispatch', False):
        return

    @wraps(determine)
    def dispatch(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens,
                 use_cascade_attn, allow_microbatching=True, force_eager=False,
                 force_uniform_decode=None, *args, **kwargs):
        if use_cascade_attn and getattr(self, '_bfla_pipeline_force_eager', False):
            force_eager = True
        width = self.uniform_decode_query_len
        decode_limit = getattr(self, '_bfla_pipeline_decode_limit', None)
        uniform_shape = num_tokens == num_reqs * width and max_num_scheduled_tokens == width
        # An explicit uniform override belongs to the native dummy/capture path.
        if force_uniform_decode is None and decode_limit is not None:
            if uniform_shape:
                force_eager = force_eager or num_tokens > decode_limit
            else:
                force_eager = force_eager or self._bfla_pipeline_force_eager
        if force_uniform_decode is None and uniform_shape:
            batch = self.input_batch
            prefixes = batch.num_computed_tokens_cpu[:num_reqs]
            prompts = batch.num_prompt_tokens[:num_reqs]
            prompt_rows = (prefixes < prompts).nonzero()[0]
            if len(prompt_rows):
                from .settings import Settings
                settings = Settings.from_env()
                for i in prompt_rows:
                    prefix, prompt = int(prefixes[i]), int(prompts[i])
                    # The GDN prefill kernel clears recycled state on the first
                    # token. A prompt chunk matching the speculative verification
                    # width also does not stage the captured spec-decode slots.
                    needs_gdn_prefill = (prefix == 0 or width > 1) and getattr(self, '_bfla_has_gdn', False)
                    if (needs_gdn_prefill or (width == 1 and settings.enabled
                            and _may_select_blocks(self, prefix, prompt, settings.config))):
                        force_uniform_decode = False
                        force_eager = force_eager or getattr(self, '_bfla_pipeline_force_eager', False)
                        break
        return determine(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens,
                         use_cascade_attn, allow_microbatching, force_eager, force_uniform_decode, *args, **kwargs)

    dispatch._bfla_prompt_dispatch = True
    runner_class._determine_batch_execution_and_padding = dispatch


def install_v2_ubatches(ubatch_class):
    original = ubatch_class.prepare
    if getattr(original, '_bfla_microbatch', False):
        return

    @wraps(original)
    def prepare(self, *args, **kwargs):
        # Both slices can touch the same request and shared execution plan on
        # different threads. The sliced InputBatch also retains whole-batch drafts.
        token = sparse_prefill_disabled.set(True)
        try:
            return original(self, *args, **kwargs)
        finally:
            sparse_prefill_disabled.reset(token)

    prepare._bfla_microbatch = True
    ubatch_class.prepare = prepare


def _batch_v2_invalidations(runner_class, name, has_updates):
    method = getattr(runner_class, name, None)
    if method is None or getattr(method, '_bfla_batch_retire', False):
        return

    @wraps(method)
    def update_requests(self, scheduler_output):
        if not has_updates(scheduler_output):
            return method(self, scheduler_output)
        state = self.model_state
        if getattr(state, '_bfla_retiring', None) is not None:
            return method(self, scheduler_output)
        retiring = state._bfla_retiring = set()
        try:
            return method(self, scheduler_output)
        finally:
            del state._bfla_retiring
            from .execution import release_requests
            release_requests(retiring)

    update_requests._bfla_batch_retire = True
    setattr(runner_class, name, update_requests)


def install_v2_runner(runner_class):
    install_v2_retirement(runner_class)
    gather = runner_class.gather_batch_req_state
    if getattr(gather, '_bfla_prompt_dispatch', False):
        return
    _install_pipeline_init(runner_class, v2=True)

    @wraps(gather)
    def gather_batch(self, scheduler_output, dummy_run):
        batch, uniform = gather(self, scheduler_output, dummy_run)
        if (batch is None or not batch.has_prefill or uniform != 1
                or batch.prefill_runs_as_decode_np is None):
            return batch, uniform
        from .settings import Settings
        settings = Settings.from_env()
        if settings.enabled:
            config = settings.config
            prompts = getattr(self.model_state, '_bfla_prompt_lengths', {})
            for i, remapped in enumerate(batch.prefill_runs_as_decode_np):
                if not remapped or batch.num_scheduled_tokens[i] != 1:
                    continue
                if batch.num_draft_tokens_np is not None and batch.num_draft_tokens_np[i]:
                    continue
                prefix = int(batch.num_computed_prefill_tokens_np[i])
                prompt = prompts.get(batch.req_ids[i], 0)
                if _may_select_blocks(self, prefix, prompt, config):
                    return batch._replace(decode_graph_eligible=False), None
        return batch, uniform

    gather_batch._bfla_prompt_dispatch = True
    runner_class.gather_batch_req_state = gather_batch


def install_v2_retirement(runner_class):
    from .kv_repair import install_v2_positions
    install_v2_positions(runner_class)
    _batch_v2_invalidations(runner_class, 'finish_requests',
                            lambda step: step.finished_req_ids or step.preempted_req_ids)
    _batch_v2_invalidations(runner_class, 'add_requests', lambda step: step.scheduled_new_reqs)
    execute = runner_class.execute_model
    if getattr(execute, '_bfla_retirement', False):
        return

    @wraps(execute)
    def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, *args, **kwargs):
        repairs = getattr(scheduler_output, 'bfla_repaired_req_ids', ())
        if repairs and not dummy_run:
            from .execution import release_requests
            release_requests(repairs)
        previous = self.execute_model_state
        result = execute(self, scheduler_output, intermediate_tensors, dummy_run, *args, **kwargs)
        step = self.execute_model_state
        if not dummy_run and step is not None and step is not previous and step.input_batch.has_prefill:
            from .execution import POOLS, release_requests
            if POOLS:
                batch = step.input_batch
                prompts = getattr(self.model_state, '_bfla_prompt_lengths', {})
                drafts = batch.num_draft_tokens_per_req
                completed = set()
                for i, key in enumerate(batch.req_ids[:batch.num_reqs]):
                    if not batch.is_prefilling_np[i]:
                        continue
                    prefix = int(batch.num_computed_prefill_tokens_np[i])
                    valid = int(batch.num_scheduled_tokens[i]) - (int(drafts[i]) if drafts is not None else 0)
                    if prefix < prompts.get(key, 0) <= prefix + valid:
                        completed.add(key)
                release_requests(completed)
        return result

    execute_model._bfla_retirement = True
    runner_class.execute_model = execute_model


def install():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    install_v1(GPUModelRunner)
    install_v2_ubatches(UBatchRunner)
    install_v2(DefaultModelState)
    install_v2(MambaHybridModelState)
    install_v2_runner(GPUModelRunnerV2)
