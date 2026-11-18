"""Graph-safe feature-only NCCL reduction; leave model collectives unchanged."""


def has_feature_reducer(group):
    """Native TP can run without PyNccl, while BFLA TP features require it."""
    if group.world_size == 1:
        return True
    communicator = getattr(group.device_communicator, 'pynccl_comm', None)
    return communicator is not None and not communicator.disabled


def feature_reducer(group):
    """Reduce index features in place on vLLM's NCCL communicator."""
    if group.world_size == 1:
        return lambda x: x
    if not has_feature_reducer(group):
        raise ValueError('active vLLM PyNccl communicator required for TP feature reduction')
    communicator = group.device_communicator.pynccl_comm
    def reduce_inplace(x):
        return communicator.all_reduce(x,x)
    return reduce_inplace
