#include <assert.h>
#include "types.hpp"
#include "torch/torch.h"
#include "torch/script.h"

torch::Tensor multiclass_nms(const torch::Tensor& multi_bboxes,
                             const torch::Tensor& multi_scores,
                             float score_thr, float iou_thr,
                             int max_num);

void batched_nms(const torch::Tensor& bboxes, const torch::Tensor& scores,
                 const torch::Tensor& idxs, torch::Tensor& keep,
                 float iou_thr, bool class_agnostic);
