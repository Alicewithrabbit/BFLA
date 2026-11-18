"""Recover synchronous external KV failures using vLLM's native retry lifecycle."""
import time
from dataclasses import dataclass
from functools import wraps


@dataclass
class _Retry:
    target: int
    first_hash: tuple | None
    generation: int
    acknowledged: int = 0


class _RepairWindow:
    """Only active on the scheduler that received a physical-page failure."""

    def __init__(self, scheduler):
        self.retries = {}
        self.steps = {}
        self.stale = {}
        self.cache_blocks = scheduler.kv_cache_manager.coordinator.cache_blocks
        self.remote_hit = scheduler.connector.get_num_new_matched_tokens
        scheduler.kv_cache_manager.coordinator.cache_blocks = self.cache_confirmed
        scheduler.connector.get_num_new_matched_tokens = self.match_remote

    def cache_confirmed(self, request, num_tokens):
        # allocate_slots normally publishes even the chunk about to execute.
        retry = self.retries.get(request)
        if retry is not None:
            if request.num_preemptions != retry.generation:
                retry.generation = request.num_preemptions
                retry.acknowledged = 0
            num_tokens = min(num_tokens, retry.acknowledged)
        return self.cache_blocks(request, num_tokens)

    def match_remote(self, request, num_local_tokens):
        # A retry may reuse valid local APC; repeating the same external hit
        # would reload the failed range, possibly forever.
        if request in self.retries:
            return 0, False
        return self.remote_hit(request, num_local_tokens)

    def join_if_shared(self, request):
        if request in self.retries:
            return
        # A new arrival can compute independently from the surviving local
        # prefix. Quarantine its publication too if it shares the failed hash.
        for retry in self.retries.values():
            if retry.first_hash:
                index, fault_hash = retry.first_hash
                if index < len(request.block_hashes) and request.block_hashes[index] == fault_hash:
                    self.retries[request] = _Retry(min(retry.target, request.num_tokens),
                                                   None, request.num_preemptions)
                    return

    def retry(self, scheduler, request, first_page, target, counts):
        key = request.request_id
        # Track the earliest hash *inside* the first dirty physical page (the
        # hash granularity can be smaller than the physical allocation).
        index = first_page * scheduler.block_size // scheduler.hash_block_size
        marker = (index, request.block_hashes[index]) if index < len(request.block_hashes) else None
        previous = self.retries.get(request)
        if previous is not None:
            target = max(target, previous.target)
            if previous.first_hash and (not marker or previous.first_hash[0] < marker[0]):
                marker = previous.first_hash
        self.retries[request] = _Retry(min(target, request.num_tokens), marker, request.num_preemptions)

        # Native drop-mode accounts for every stale row of every outstanding
        # step, including the result currently being processed.
        remaining = request.num_in_flight_tokens - counts.get(key, 0)
        if remaining:
            old = self.stale.get(key)
            pins = old[2] if old else []
            if not scheduler.defer_block_free:
                # Production overlapping consumers use native deferred frees.
                # Also protect connectors configured without that fence.
                (blocks,) = scheduler.kv_cache_manager.get_blocks(key).blocks
                blocks = [block for block in blocks if not block.is_null]
                scheduler.kv_cache_manager.block_pool.touch(blocks)
                pins.extend(blocks)
            self.stale[key] = (request, remaining, pins)

    def submitted(self, scheduler, step):
        # Only recovering rows carry a CPU-side receipt. The scheduler keeps
        # the original SchedulerOutput paired with its ordered executor future.
        records = {}
        for request, retry in self.retries.items():
            key = request.request_id
            if key in step.num_scheduled_tokens and scheduler.requests.get(key) is request:
                end = request.num_computed_tokens - len(step.scheduled_spec_decode_tokens.get(key, ()))
                records[key] = request, request.num_preemptions, retry, min(end, request.num_tokens)
        if records:
            self.steps[id(step)] = records

    def receiving(self, scheduler, step):
        records = self.steps.pop(id(step), {})
        for key in self.stale.keys() & step.num_scheduled_tokens.keys():
            old, remaining, pins = self.stale[key]
            count = step.num_scheduled_tokens[key]
            assert count <= remaining
            if count == remaining:
                scheduler.kv_cache_manager.block_pool.free_blocks(reversed(pins))
                del self.stale[key]
            else:
                self.stale[key] = old, remaining - count, pins
            records[key] = (old, None, None, 0)
        for key, (old, generation, retry, end) in records.items():
            current = scheduler.requests.get(key)
            if (retry is not None and current is old and old.num_preemptions == generation
                    and self.retries.get(old) is retry and not old.num_stale_output_tokens):
                retry.acknowledged = max(retry.acknowledged, end)

    def prune(self, scheduler):
        for request, retry in tuple(self.retries.items()):
            if (scheduler.requests.get(request.request_id) is not request or request.is_finished()
                    or retry.acknowledged >= retry.target):
                del self.retries[request]
        if not (self.retries or self.stale or self.steps):
            scheduler.kv_cache_manager.coordinator.cache_blocks = self.cache_blocks
            scheduler.connector.get_num_new_matched_tokens = self.remote_hit
            del scheduler._bfla_kv_repair_window


def _dirty_suffixes(scheduler, invalid_ids, retired):
    """Close over resident and recently retired rows sharing a physical page."""
    manager = scheduler.kv_cache_manager
    null_id = manager.block_pool.null_block.block_id
    dirty = set(invalid_ids) - {null_id}
    live = [*scheduler.running, *scheduler.kv_holding_waiting]
    # A connector can retain the allocation of a finished request after it
    # leaves both queues. Its downstream cache entries still need eviction.
    live.extend(request for request in scheduler.requests.values() if request.is_finished())
    tables = [(req, req.num_computed_tokens, manager.get_block_ids(req.request_id)[0])
              for req in live]
    tables.extend((req, target, [block.block_id for block in blocks])
                  for req, target, blocks in retired)
    starts = [len(blocks) for _, _, blocks in tables]
    while True:
        previous_size = len(dirty)
        for idx, (_, _, blocks) in enumerate(tables):
            stop = starts[idx]
            first = next((i for i in range(stop) if blocks[i] in dirty), None)
            if first is not None:
                starts[idx] = first
                dirty.update(blocks[first:stop])
                dirty.discard(null_id)
        if len(dirty) == previous_size:
            break
    recover = {}
    for (req, target, blocks), first in zip(tables, starts):
        if first == len(blocks) or req.is_finished() or scheduler.requests.get(req.request_id) is not req:
            continue
        previous_start, previous_target = recover.get(req, (first, target))
        recover[req] = min(first, previous_start), max(target, previous_target)
    return recover, dirty


def install_scheduler(scheduler_class):
    original_handle = scheduler_class._handle_invalid_blocks
    if getattr(original_handle, '_bfla_kv_repair', False):
        return
    original_schedule = scheduler_class.schedule
    original_free = scheduler_class._free_request_blocks
    original_finish = scheduler_class._free_request
    original_update = scheduler_class.update_from_output
    original_ready = scheduler_class._handle_blocked_waiting_request

    @wraps(original_handle)
    def handle(self, invalid_block_ids, num_scheduled_tokens):
        if not self.recompute_kv_load_failures or len(self.kv_cache_config.kv_cache_groups) != 1:
            return original_handle(self, invalid_block_ids, num_scheduled_tokens)
        recover, dirty = _dirty_suffixes(self, invalid_block_ids,
                                        getattr(self, '_bfla_reporting_retired', ()))
        affected = original_handle(self, invalid_block_ids, num_scheduled_tokens)
        # Permanently unlink the old allocations, including aliases belonging
        # to aborted requests. Reference counts still guard in-flight writes.
        self.kv_cache_manager.evict_blocks(dirty)
        if not recover:
            return affected
        window = getattr(self, '_bfla_kv_repair_window', None)
        if window is None:
            window = self._bfla_kv_repair_window = _RepairWindow(self)
        from vllm.v1.request import RequestStatus
        for request, (first, target) in reversed(recover.items()):
            key = request.request_id
            window.retry(self, request, first, target, num_scheduled_tokens)
            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
                self._preempt_request(request, time.monotonic(), drop_stale_output=True)
                self.prev_step_scheduled_req_ids.discard(key)
                affected.add(key)
            elif request.status == RequestStatus.PREEMPTED:
                # Native memory-pressure preemption normally delivers its
                # in-flight samples. Its retired allocation proved them stale.
                request.drop_stale_output = True
                self.prev_step_scheduled_req_ids.discard(key)
                affected.add(key)
            elif request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # The connector still owns the destination until recv finishes.
                request.num_computed_tokens = 0
                self.failed_recving_kv_req_ids.add(key)
            else:
                # A waiting streaming session keeps its token history. Release
                # its poisoned pages on its next admission, after any stream
                # update has consumed the previous committed frontier.
                request._bfla_release_waiting_kv = True
        return affected

    @wraps(original_ready)
    def ready(self, request):
        if not original_ready(self, request):
            return False
        window = getattr(self, '_bfla_kv_repair_window', None)
        if window is None:
            return True
        if getattr(request, '_bfla_release_waiting_kv', False):
            self._free_request_blocks(request)
            request.num_computed_tokens = 0
            del request._bfla_release_waiting_kv
        window.join_if_shared(request)
        return True

    @wraps(original_free)
    def free(self, request):
        loads = getattr(self, '_bfla_pending_sync_loads', None)
        if loads and request.num_in_flight_tokens:
            # An abort/preemption can remove the only block table describing
            # the corrupt suffix before the connector replies. Pin its identity
            # only through sync loads already outstanding when it was removed.
            (blocks,) = self.kv_cache_manager.get_blocks(request.request_id).blocks
            blocks = tuple(blocks)
            if blocks:
                for retired in loads.values():
                    self.kv_cache_manager.block_pool.touch(blocks)
                    retired.append((request, request.num_computed_tokens, blocks))
                # Without a native fence, a later compute-only batch can
                # still write these blocks after the sync load has reported.
                last = self._bfla_last_writer_step
                if not self.defer_block_free and last not in loads:
                    self.kv_cache_manager.block_pool.touch(blocks)
                    self.__dict__.setdefault('_bfla_write_fences', {}).setdefault(last, []).append(blocks)
        return original_free(self, request)

    @wraps(original_finish)
    def finish(self, request, *args, **kwargs):
        if request.num_in_flight_tokens and (
                getattr(self, '_bfla_pending_sync_loads', None) or
                getattr(self, '_bfla_kv_repair_window', None) or
                request.request_id in getattr(self, '_bfla_departed', ())):
            # The engine can reuse a public request ID before all of this
            # incarnation's FIFO results return. Its successor owns no part of
            # that accounting, even when the load ultimately succeeds.
            departed = self.__dict__.setdefault('_bfla_departed', {})
            departed.setdefault(request.request_id, []).append([request, request.num_in_flight_tokens])
        return original_finish(self, request, *args, **kwargs)

    @wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        window = getattr(self, '_bfla_kv_repair_window', None)
        if window is not None:
            window.prune(self)
        step = original_schedule(self, *args, **kwargs)
        if (step.has_sync_kv_loads and self.recompute_kv_load_failures
                and len(self.kv_cache_config.kv_cache_groups) == 1):
            self.__dict__.setdefault('_bfla_pending_sync_loads', {})[id(step)] = []
        if step.total_num_scheduled_tokens and getattr(self, '_bfla_pending_sync_loads', None):
            self._bfla_last_writer_step = id(step)
        if window is not None and window.retries:
            window.submitted(self, step)
        return step

    @wraps(original_update)
    def update(self, scheduler_output, model_runner_output):
        window = getattr(self, '_bfla_kv_repair_window', None)
        if window is not None:
            window.receiving(self, scheduler_output)
        loads = getattr(self, '_bfla_pending_sync_loads', None)
        retired = loads.pop(id(scheduler_output), ()) if loads else ()
        if loads is not None and not loads:
            del self._bfla_pending_sync_loads
        departed = getattr(self, '_bfla_departed', None)
        restore_delivery = []
        if departed:
            for key in departed.keys() & scheduler_output.num_scheduled_tokens.keys():
                old, remaining = departed[key][0]
                count = scheduler_output.num_scheduled_tokens[key]
                current = self.requests.get(key)
                if current is not old:
                    old.num_in_flight_tokens -= count
                    old.num_stale_output_tokens = max(0, old.num_stale_output_tokens - count)
                if current is not None and current is not old:
                    if current.num_stale_output_tokens and not current.drop_stale_output:
                        retry = window.retries.get(current) if window else None
                        restore_delivery.append((current, current.num_preemptions, retry))
                    current.num_in_flight_tokens += count
                    current.num_stale_output_tokens += count
                    current.drop_stale_output = True
                remaining -= count
                assert remaining >= 0
                if remaining:
                    departed[key][0][1] = remaining
                else:
                    departed[key].pop(0)
                    if not departed[key]:
                        del departed[key]
            if not departed:
                del self._bfla_departed
        if retired:
            self._bfla_reporting_retired = retired
        try:
            result = original_update(self, scheduler_output, model_runner_output)
        finally:
            # Suppress only the departed incarnation's row. Its successor may
            # have separately been preempted with deliverable output pending.
            window = getattr(self, '_bfla_kv_repair_window', None)
            for request, generation, previous_retry in restore_delivery:
                retry = window.retries.get(request) if window else None
                if request.num_preemptions == generation and retry is previous_retry:
                    request.drop_stale_output = False
            if retired:
                del self._bfla_reporting_retired
                for _, _, blocks in retired:
                    self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
            fences = getattr(self, '_bfla_write_fences', None)
            if fences:
                for blocks in fences.pop(id(scheduler_output), ()):
                    self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
                if not fences:
                    del self._bfla_write_fences
            if getattr(self, '_bfla_last_writer_step', None) == id(scheduler_output):
                del self._bfla_last_writer_step
        window = getattr(self, '_bfla_kv_repair_window', None)
        if window is not None:
            window.prune(self)
        return result

    handle._bfla_kv_repair = True
    scheduler_class._free_request_blocks = free
    scheduler_class._free_request = finish
    scheduler_class._handle_invalid_blocks = handle
    scheduler_class._handle_blocked_waiting_request = ready
    scheduler_class.schedule = schedule
    scheduler_class.update_from_output = update


def install_v2_positions(runner_class):
    update = getattr(runner_class, 'update_requests', None)
    if update is None or getattr(update, '_bfla_repair_positions', False):
        return

    @wraps(update)
    def update_requests(self, scheduler_output):
        result = update(self, scheduler_output)
        rewound = getattr(scheduler_output, 'bfla_rewound_req_ids', ())
        if rewound:
            # vLLM applies the authoritative cached-request prefix to NumPy in
            # update_requests, but normally advances GPU positions by sampling.
            # A synchronous load failure can move a running prefill backwards.
            states = self.req_states
            changed = False
            for key in rewound:
                slot = states.req_id_to_index[key]
                prefix = int(states.num_computed_tokens_np[slot])
                # Readmission can replay generated tokens after the original
                # prompt. At/after prefill_len the CPU count is optimistic and
                # the device already accounts for speculative rejections.
                if prefix < states.prefill_len.np[slot]:
                    states.num_computed_tokens.stage_write_elem(slot, prefix)
                    changed = True
            if changed:
                states.num_computed_tokens.apply_write()
        return result

    update_requests._bfla_repair_positions = True
    runner_class.update_requests = update_requests
