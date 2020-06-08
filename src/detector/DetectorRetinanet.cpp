#include "DetectorRetinanet.hpp"

DetectorRetinanet::DetectorRetinanet()
{

}

DetectorRetinanet::~DetectorRetinanet() {

}

void DetectorRetinanet::LoadParams(const Params& params, torch::DeviceType* device_type) {

    DetectorCommon::LoadParams(params, device_type);

    retina_head_params_ = params.retina_head_params_;

    int octave_base_scale_ = retina_head_params_.octave_base_scale_;
    int scales_per_octave_ = retina_head_params_.scales_per_octave_;

    std::vector<float> anchor_scales;
    for(int i = 0; i < scales_per_octave_; i++) {
        anchor_scales.push_back((pow(2,(float(i) / float(scales_per_octave_)))) * octave_base_scale_);
    }
    get_anchor_generators({}, anchor_scales, retina_head_params_.anchor_ratios_);
}

void DetectorRetinanet::LoadTracedModule(){
   DetectorCommon::LoadTracedModule();
}

void DetectorRetinanet::Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes){

  DetectorCommon::DetectOneStage(image, detected_boxes);
}
