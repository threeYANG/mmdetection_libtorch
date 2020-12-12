#include "SingleRoIExtractor.hpp"

SingleRoIExtractor::SingleRoIExtractor()
{

}

void SingleRoIExtractor::init_params(const RoiExtractorParams& roi_extractor_params){
    out_size_ = roi_extractor_params.out_size_;
    out_channels_ = roi_extractor_params.out_channels_;
    featmap_strides_ = roi_extractor_params.featmap_strides_;
    sampling_ratio_ = roi_extractor_params.sampling_ratio_;
    finest_scale_ = 56; //Default: 56
    roi_scale_factor_ = 0; // not support roi_scale_factor > 0;

    build_roi_align_layer();
}

SingleRoIExtractor::~SingleRoIExtractor()
{

}

void SingleRoIExtractor::build_roi_align_layer() {
    for(int i = 0; i < featmap_strides_.size(); i++) {
        float spatial_scale = 1.0 / float(featmap_strides_[i]);
        roi_align_layers_.push_back(RoIAlign(out_size_, spatial_scale, sampling_ratio_));
    }
}



/*****
      Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
**********/

torch::Tensor SingleRoIExtractor::map_roi_levels(const torch::Tensor& rois, int num_levels) {
    torch::Tensor scale = torch::sqrt((rois.select(1, 3) - rois.select(1, 1)) * (rois.select(1, 4) - rois.select(1, 2)));
    torch::Tensor target_lvls = torch::floor(torch::log2(scale / float(finest_scale_) + 1.0e-6 ));
    target_lvls.clamp_(0, num_levels-1);
    return target_lvls;
}

torch::Tensor SingleRoIExtractor::bbox_roi_extractor(const std::vector<torch::Tensor>& feats, const torch::Tensor& rois) {
   torch::Tensor target_lvls = map_roi_levels(rois, featmap_strides_.size());
   torch::Tensor roi_feats = feats[0].new_zeros({rois.size(0), out_channels_, out_size_, out_size_});
   for(int i =0 ; i < featmap_strides_.size(); i++)
   {
       std::vector<torch::Tensor> index = torch::where(target_lvls == i);
       torch::Tensor rois_ = rois.index_select(0, index[0]);
       int num_rois = rois_.size(0);
       if (num_rois == 0){
           continue;
       }
       torch::Tensor output = roi_align_layers_[i].roi_align_forward_cuda(feats[i], rois_);
       roi_feats.index_copy_(0, index[0], output);
   }
   return roi_feats;
}
