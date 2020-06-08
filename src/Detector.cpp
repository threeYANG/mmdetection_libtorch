//
// Created by dl on 2020/1/2.
//
#include <memory>
#include "Detector.hpp"
#include "DetectorImpl.hpp"




Detector::Detector() {
   impl_ = nullptr;
}

Detector::~Detector() {

}

long Detector::Create(DetetorType detetorType){
    std::unique_ptr<DetectorImpl> imp_temp = DetectorImpl::Create(detetorType);
    impl_ = std::move(imp_temp);
    if (impl_ == nullptr) {
        std::cout << "detector create failed!" << std::endl;
        return -2;
    }
    return 0;
}

void Detector::LoadParams(const Params& params, torch::DeviceType* device_type) {
    impl_->LoadParams(params, device_type);
}


void Detector::LoadTracedModule() {
    if(impl_ != nullptr) {
        impl_->LoadTracedModule();
    } else {
        std::cout << "impl is nullptr, LoadTracedModule failed" << std::endl;
    }
}


long Detector::Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes){
    if(impl_ != nullptr) {
        impl_->Detect(image, detected_boxes);
        return 0;
    } else {
        std::cout << "impl is nullptr, Detect failed" << std::endl;
    }
    return -1;
}
