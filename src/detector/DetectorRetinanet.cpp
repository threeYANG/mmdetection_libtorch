#include "DetectorRetinanet.hpp"

DetectorRetinanet::DetectorRetinanet()
{

}

DetectorRetinanet::~DetectorRetinanet() {

}

void DetectorRetinanet::LoadParams(const Params& params, torch::DeviceType* device_type) {

    SingleStage::LoadParams_Single(params, device_type);

    in_channels_ = params.retinanet_params_.in_channels_;
    stacked_convs_ = params.retinanet_params_.stacked_convs_;
    feat_channels_ = params.retinanet_params_.feat_channels_;
    octave_base_scale_ = params.retinanet_params_.octave_base_scale_;
    scales_per_octave_ = params.retinanet_params_.scales_per_octave_;
    anchor_ratios_ = params.retinanet_params_.anchor_ratios_;

    std::vector<float> anchor_scales;
    for(int i = 0; i < scales_per_octave_; i++) {
        anchor_scales.push_back((pow(2,(float(i) / float(scales_per_octave_)))) * octave_base_scale_);
    }
    get_anchor_generators({}, anchor_scales, anchor_ratios_);
}

void DetectorRetinanet::LoadTracedModule(){
   LoadTracedModule_Single();
}

void DetectorRetinanet::Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes){

  Detect_Single(image, detected_boxes);
}
