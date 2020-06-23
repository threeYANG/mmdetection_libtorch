#include "torch/torch.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <iostream>
#include "RoIAlign.hpp"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(const scalar_t *bottom_data,
                                         const int height, const int width,
                                         scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;
  // do bilinear interpolation
  scalar_t lt = bottom_data[y_low * width + x_low];
  scalar_t rt = bottom_data[y_low * width + x_high];
  scalar_t lb = bottom_data[y_high * width + x_low];
  scalar_t rb = bottom_data[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

  return val;
}

template <typename scalar_t>
__global__ void ROIAlignForwardV1(
    const int nthreads, const scalar_t *bottom_data,
    const scalar_t *bottom_rois, const scalar_t spatial_scale,
    const int sample_num, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the aligned output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = (offset_bottom_rois[3] + 1) * spatial_scale;
    scalar_t roi_end_h = (offset_bottom_rois[4] + 1) * spatial_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);

    scalar_t bin_size_h = roi_height / pooled_height;
    scalar_t bin_size_w = roi_width / pooled_width;

    const scalar_t *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    int sample_num_h = (sample_num > 0)
                           ? sample_num
                           : ceil(roi_height / pooled_height);  // e.g., = 2
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    scalar_t output_val = 0;
    for (int iy = 0; iy < sample_num_h; iy++) {
      const scalar_t y = roi_start_h + ph * bin_size_h +
                         (scalar_t)(iy + scalar_t(.5f)) * bin_size_h /
                             (scalar_t)(sample_num_h);
      for (int ix = 0; ix < sample_num_w; ix++) {
        const scalar_t x = roi_start_w + pw * bin_size_w +
                           (scalar_t)(ix + scalar_t(.5f)) * bin_size_w /
                               (scalar_t)(sample_num_w);
        scalar_t val = bilinear_interpolate<scalar_t>(offset_bottom_data,
                                                      height, width, y, x);
        output_val += val;
      }
    }
    output_val /= (sample_num_h * sample_num_w);
    top_data[index] = output_val;
  }
}

int ROIAlignForwardLaucher(const torch::Tensor features, const torch::Tensor rois,
                           const float spatial_scale, const int sample_num,
                           const int channels, const int height,
                           const int width, const int num_rois,
                           const int pooled_height, const int pooled_width,
                           torch::Tensor output) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "ROIAlignLaucherForward", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();

        ROIAlignForwardV1<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0,
            at::cuda::getCurrentCUDAStream()>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                sample_num, channels, height, width, pooled_height,
                pooled_width, top_data);
      }));
  //std::cout<<cudaGetLastError()<<std::endl;
  //THCudaCheck(cudaGetLastError());
  return 1;
}
