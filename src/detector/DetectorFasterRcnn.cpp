#include "DetectorFasterRcnn.hpp"

DetectorFasterRcnn::DetectorFasterRcnn()
{

}

DetectorFasterRcnn::~DetectorFasterRcnn(){

}

void DetectorFasterRcnn::LoadParams(const Params& params, torch::DeviceType* device_type){
    DetectorCommon::LoadParams(params, device_type);
    get_anchor_generators({}, rpn_head_params_.anchor_scales_, rpn_head_params_.anchor_ratios_);
}


void DetectorFasterRcnn::LoadTracedModule() {
    DetectorCommon::LoadTracedModule();
}


void DetectorFasterRcnn::Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) {

   DetectTwoStage(image, detected_boxes);
}
