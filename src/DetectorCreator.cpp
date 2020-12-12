//
// Created by dl on 2020/1/3.
//

#include "DetectorImpl.hpp"
#include "DetectorSSD.hpp"
#include "DetectorFCOS.hpp"
#include "DetectorRetinanet.hpp"
#include "DetectorFasterRcnn.hpp"

std::unique_ptr<DetectorImpl> DetectorCreator::create_detector(DetetorType detector_type) {
    switch(detector_type) {
        case DetetorType ::SSD:
            std::cout << " successfully create SSD detector" << std::endl;
            return std::unique_ptr<DetectorImpl>(new DetectorSSD());
    case DetetorType ::Retinanet:
        std::cout << " successfully create Retinanet detector" << std::endl;
        return std::unique_ptr<DetectorImpl>(new DetectorRetinanet());
    case DetetorType::FasterRcnn:
        std::cout << " successfully create FasterRcnn detector" << std::endl;
        return std::unique_ptr<DetectorImpl>(new DetectorFasterRcnn());
    case DetetorType::FCOS:
        std::cout << " successfully create FCOS detector" << std::endl;
        return std::unique_ptr<DetectorImpl>(new DetectorFCOS());
    default:
            std::cout << "only support SSD, Retinanet, FasterRcnn, FCOS now" << std::endl;
    }
    return nullptr;
}
