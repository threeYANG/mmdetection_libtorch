#include "RoIAlign.hpp"


RoIAlign::RoIAlign(int out_size, float spatial_scale, int sampling_ratio,
                   std::string pool_mode,
                   bool use_torchvision, bool aligned)
{
   aligned_height_ = out_size;
   aligned_width_ = out_size;
   spatial_scale_ = spatial_scale;
   use_torchvision_ = use_torchvision; // only support false
   aligned_ = aligned;
   sampling_ratio_ = sampling_ratio; //default
   pool_mode_ = pool_mode == "avg"? 1 : 0;
}

RoIAlign::~RoIAlign() {

}


torch::Tensor RoIAlign::roi_align_forward_cuda(const Tensor& input,
                                               const Tensor& rois) {

    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(rois);
    IntArrayRef output_shape = {rois.size(0), input.size(1), aligned_height_, aligned_width_};
    torch::Tensor output = input.new_zeros(output_shape);
    CHECK_CUDA_INPUT(output);

    torch::Tensor argmax_y, argmax_x;
    if (pool_mode_ == 0) {
        argmax_y = input.new_zeros(output_shape);
        argmax_x = input.new_zeros(output_shape);
    } else {
       argmax_y = input.new_zeros(0);
       argmax_x = input.new_zeros(0);
    }
    CHECK_CUDA_INPUT(argmax_y);
    CHECK_CUDA_INPUT(argmax_x);

    ROIAlignForwardCUDAKernelLauncher(
      input, rois, output, argmax_y, argmax_x, aligned_height_, aligned_width_,
      spatial_scale_, sampling_ratio_, pool_mode_, aligned_);
    return output;
}






