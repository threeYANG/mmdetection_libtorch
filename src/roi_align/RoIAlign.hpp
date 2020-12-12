#ifndef ROIALIGN_HPP
#define ROIALIGN_HPP

#include <ATen/ATen.h>

#include <cmath>
#include <vector>
#include "torch/torch.h"
#include "torch/script.h"

#include "pytorch_cpp_helper.hpp"


void ROIAlignForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                       Tensor argmax_y, Tensor argmax_x,
                                       int aligned_height, int aligned_width,
                                       float spatial_scale, int sampling_ratio,
                                       int pool_mode, bool aligned);




class RoIAlign
{
public:
    RoIAlign(int out_size, float spatial_scale, int sampling_ratio,
             std::string pool_mode = "avg",
             bool use_torchvision = false, bool aligned = true);
    ~RoIAlign();

    torch::Tensor roi_align_forward_cuda(const Tensor& input, const Tensor& rois);

private:
    int aligned_height_;
    int aligned_width_;
    float spatial_scale_;
    int sampling_ratio_;
    int pool_mode_;
    bool use_torchvision_;
    bool aligned_;
};

#endif // ROIALIGN_HPP
