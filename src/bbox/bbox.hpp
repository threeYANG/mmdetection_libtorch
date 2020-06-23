//
// Created by dl on 2020/4/15.
//
#include <assert.h>
#include "types.hpp"
#include "torch/torch.h"
#include "torch/script.h"


extern "C" {
at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);
}

torch::Tensor delta2bbox(const torch::Tensor& anchors, const torch::Tensor& deltas,
                         const std::vector<int>& max_shape,
                         const std::vector<float>& means = {0.0, 0.0, 0.0, 0.0},
                         const std::vector<float>& stds = {1.0, 1.0, 1.0, 1.0},
                         float wh_ratio_clip = 16.0 / 1000.0);



torch::Tensor multiclass_nms(const torch::Tensor& multi_bboxes,
                             const torch::Tensor& multi_scores,
                             float score_thr, float iou_thr,
                             int max_num=-1);

torch::Tensor singleclass_nms(const torch::Tensor& proposals,float iou_thr);

void bbox2result(torch::Tensor& result, float thresh, cv::Size2f scale,
                 std::vector<DetectedBox>& detected_boxes);

torch::Tensor bbox2roi(const torch::Tensor& proposals);
