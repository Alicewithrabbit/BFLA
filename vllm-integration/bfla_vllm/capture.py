from contextlib import contextmanager

@contextmanager
def capture_context(stream):
    """Use vLLM's communicator buffer registration on the existing stream."""
    from vllm.distributed.parallel_state import get_tp_group,get_pp_group,GraphCaptureContext
    if get_tp_group().world_size==1:
        yield
    else:
        context=GraphCaptureContext(stream)
        with get_tp_group().graph_capture(context),get_pp_group().graph_capture(context):
            yield

