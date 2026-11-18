#pragma once

#include <ATen/ATen.h>

namespace bfla {

void launch_indexer_key_feature(const at::Tensor& key,
                                const at::Tensor& out,
                                int query_heads,
                                int index_q_heads,
                                float eps);

}  // namespace bfla
