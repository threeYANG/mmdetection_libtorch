#ifndef SINGLEROIEXTRACTOR_H
#define SINGLEROIEXTRACTOR_H

#include <vector>
#include "torch/torch.h"
#include "torch/script.h"
#include "types.hpp"
#include "RoIAlign.hpp"

class SingleRoIExtractor
{
public:
    SingleRoIExtractor();
    ~SingleRoIExtractor();

    void init_params(const RoiExtractorParams& roi_extractor_params);
    torch::Tensor bbox_roi_extractor(const std::vector<torch::Tensor>& feats, const torch::Tensor& rois);
private:
    torch::Tensor map_roi_levels(const torch::Tensor rois, int num_levels);
    void build_roi_align_layer();
private:
    int out_size_;
    int out_channels_;
    int finest_scale_;
    int sample_num_;
    std::vector<int> featmap_strides_;
    std::vector<RoIAlign> roi_align_layers_;

};

#endif // SINGLEROIEXTRACTOR_H
