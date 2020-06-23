#include "RoIAlign.hpp"


RoIAlign::RoIAlign(int out_size, float spatial_scale, int sample_num,
                   bool use_torchvision, bool aligned)
{
   pooled_height_ = out_size;
   pooled_width_ = out_size;
   spatial_scale_ = spatial_scale;
   sample_num_ = sample_num;
   use_torchvision_ = use_torchvision;
   aligned_ = aligned;
}

RoIAlign::~RoIAlign() {

}

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int RoIAlign::roi_align_forward_cuda(at::Tensor features, at::Tensor rois,
                                     at::Tensor output) {
  CHECK_INPUT(features);
  CHECK_INPUT(rois);
  CHECK_INPUT(output);

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 5) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);

  ROIAlignForwardLaucher(features, rois, spatial_scale_, sample_num_,
                         num_channels, data_height, data_width, num_rois,
                         pooled_height_, pooled_width_, output);

  return 1;
}



