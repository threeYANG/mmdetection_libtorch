//
// Created by dl on 2020/4/15.
//
#include <assert.h>
#include "types.hpp"
#include "torch/torch.h"
#include "torch/script.h"


torch::Tensor distance2bbox(const torch::Tensor points,
                            const torch::Tensor distance,
                            const std::vector<int>& max_shape);

torch::Tensor delta2bbox(const torch::Tensor& anchors, const torch::Tensor& deltas,
                         const std::vector<int>& max_shape,
                         const std::vector<float>& means = {0.0, 0.0, 0.0, 0.0},
                         const std::vector<float>& stds = {1.0, 1.0, 1.0, 1.0},
                         float wh_ratio_clip = 16.0 / 1000.0);

void bbox2result(torch::Tensor& result, torch::Tensor& segm_results, float thresh,
                 std::vector<DetectedBox>& detected_boxes);

torch::Tensor bbox2roi(const torch::Tensor& proposals);
