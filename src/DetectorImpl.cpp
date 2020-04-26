//
// Created by dl on 2020/1/2.
//

#include "DetectorImpl.hpp"


DetectorImpl::DetectorImpl() = default;

DetectorImpl::~DetectorImpl() = default;

std::unique_ptr<DetectorImpl> DetectorImpl::Create(DetetorType detector_type) {

    return DetectorCreator::create_detector(detector_type);
}


