#include "SingleRoIExtractor.hpp"

SingleRoIExtractor::SingleRoIExtractor()
{

}

void SingleRoIExtractor::init_params(const RoiExtractorParams& roi_extractor_params){
    out_size_ = roi_extractor_params.out_size_;
    out_channels_ = roi_extractor_params.out_channels_;
    featmap_strides_ = roi_extractor_params.featmap_strides_;
    sample_num_ = roi_extractor_params.sample_num_;
    finest_scale_ = 56;

    build_roi_align_layer();
}

SingleRoIExtractor::~SingleRoIExtractor()
{

}

void SingleRoIExtractor::build_roi_align_layer() {
    for(int i = 0; i < featmap_strides_.size(); i++) {
        float spatial_scale = 1.0 / float(featmap_strides_[i]);
        roi_align_layers_.push_back(RoIAlign(out_size_, spatial_scale, sample_num_));
    }
}
torch::Tensor SingleRoIExtractor::map_roi_levels(const torch::Tensor rois, int num_levels) {
    torch::Tensor scale = torch::sqrt((rois.select(1, 3) - rois.select(1, 1) + 1) * (rois.select(1, 4) - rois.select(1, 2) + 1));
    torch::Tensor target_lvls = torch::floor(torch::log(scale / float(finest_scale_) + 1.0e-6 ));
    target_lvls.clamp_(0,num_levels-1);
    return target_lvls;
}

torch::Tensor SingleRoIExtractor::bbox_roi_extractor(const std::vector<torch::Tensor>& feats, const torch::Tensor& rois) {
   torch::Tensor target_lvls = map_roi_levels(rois, featmap_strides_.size());
   torch::Tensor roi_feats = feats[0].new_zeros({rois.size(0), out_channels_, out_size_, out_size_});

   for(int i =0 ; i < feats.size() - 1; i++)
   {
       std::vector<torch::Tensor> index = torch::where(target_lvls == i);
       torch::Tensor rois_ = rois.index_select(0, index[0]);
       int num_rois = rois_.size(0);
       if (num_rois == 0){
           continue;
       }
       int num_channels = feats[i].size(1);

       torch::Tensor output = feats[i].new_zeros({num_rois, num_channels, out_size_,out_size_});
       roi_align_layers_[i].roi_align_forward_cuda(feats[i], rois_, output);
       roi_feats.index_select(0, index[0]) = output;
   }

   return roi_feats;
}
