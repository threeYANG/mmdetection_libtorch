#ifndef SINGLEROIEXTRACTOR_H
#define SINGLEROIEXTRACTOR_H

#include <vector>
#include "torch/torch.h"
#include "torch/script.h"
#include "types.hpp"
#include "RoIAlign.hpp"


/*******
Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
***********/

class SingleRoIExtractor
{
public:
    SingleRoIExtractor();
    ~SingleRoIExtractor();

    void init_params(const RoiLayerParams& roi_layer);
    torch::Tensor roi_extractor(const std::vector<torch::Tensor>& feats, const torch::Tensor& rois);
private:
    torch::Tensor map_roi_levels(const torch::Tensor& rois, int num_levels);
    void build_roi_align_layer();
private:
    int out_size_;
    int out_channels_;
    int sampling_ratio_;
    int finest_scale_;
    int roi_scale_factor_;
    std::vector<int> featmap_strides_;
    std::vector<RoIAlign> roi_align_layers_;

};

#endif // SINGLEROIEXTRACTOR_H
