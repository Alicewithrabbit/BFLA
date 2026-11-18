#include <torch/extension.h>
#include "indexer_key_feature.h"

// Key-feature binding; the CUDA calculation is unchanged.
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("key_feature", &bfla::launch_indexer_key_feature);
}
