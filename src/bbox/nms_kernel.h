#include "torch/torch.h"
#include "torch/script.h"

extern "C" {
at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);
}

