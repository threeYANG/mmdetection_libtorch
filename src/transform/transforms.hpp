#include "types.hpp"
#include "torch/torch.h"
#include "torch/script.h"

void transform(const cv::Mat& image, torch::Tensor& image_tensor,
               const TransformParams& transform_params,
               int& net_width, int& net_height,
               torch::DeviceType* device);
