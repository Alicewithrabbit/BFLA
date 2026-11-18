"""Feature implementation dispatch by geometry, not model identity."""


class FusedFeatures:
    def __init__(self, layout):
        from . import build
        self.layout = layout
        self.warp = build.key_feature() if (layout.head_dim, layout.local_hkv) == (256, 4) else None
        self.extension = build.feature_extension() if self.warp is None else None
        if layout.head_dim == 256:
            from ._kernels.query_d256 import query_features
        else:
            from ._kernels.query_d512 import query_features
        self.query = query_features

    def key(self, key, out):
        import torch
        active = out[:key.shape[0]]
        if self.warp is None:
            self.extension.key_feature(key, active, self.layout.global_hq, 1, 1.e-6)
        else:
            self.warp.forward(key.data_ptr(), active.data_ptr(), key.shape[0], key.stride(0), key.stride(1),
                              torch.cuda.current_stream(key.device).cuda_stream)
        return active

    def query_batch(self, query, out, spans):
        for source, rows, destination in spans:
            self.query(query[source:source+rows], out[destination:destination+rows])


def make_features(layout, key_rows, query_rows, all_reduce, config, device):
    if layout.tp_size > 1:
        if all_reduce is None:
            raise ValueError('TP feature construction requires an all_reduce callback')
        from ._kernels.distributed import make_features as distributed
        return distributed(layout, key_rows, query_rows, all_reduce, config, device)
    if config.index_dim == 128:
        return FusedFeatures(layout)
    from ._kernels.features import GeneralizedFeatures
    return GeneralizedFeatures(layout, config.index_dim)
