#ifndef ROIALIGN_HPP
#define ROIALIGN_HPP

#include <ATen/ATen.h>

#include <cmath>
#include <vector>
#include "torch/torch.h"
#include "torch/script.h"


extern "C" {
int ROIAlignForwardLaucher(const at::Tensor features, const at::Tensor rois,
                           const float spatial_scale, const int sample_num,
                           const int channels, const int height,
                           const int width, const int num_rois,
                           const int pooled_height, const int pooled_width,
                           at::Tensor output);
}


extern "C" {
at::Tensor ROIAlignForwardV2Laucher(const at::Tensor& input,
                                    const at::Tensor& rois,
                                    const float spatial_scale,
                                    const int pooled_height,
                                    const int pooled_width,
                                    const int sampling_ratio, bool aligned);

}

class RoIAlign
{
public:
    RoIAlign(int out_size, float spatial_scale, int sample_num,
             bool use_torchvision = false, bool aligned = false);
    ~RoIAlign();

    int roi_align_forward_cuda(at::Tensor features, at::Tensor rois,
                               at::Tensor output);

private:
    int pooled_height_;
    int pooled_width_;
    float spatial_scale_;
    bool aligned_;
    int sample_num_;
    bool use_torchvision_;
};

#endif // ROIALIGN_HPP
