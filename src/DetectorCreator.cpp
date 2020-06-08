//
// Created by dl on 2020/1/3.
//

#include "DetectorImpl.hpp"
#include "DetectorSSD.hpp"
#include "DetectorRetinanet.hpp"
#include "DetectorFasterRcnn.hpp"

std::unique_ptr<DetectorImpl> DetectorCreator::create_detector(DetetorType detector_type) {
    switch(detector_type) {
        case DetetorType ::SSD:
            std::cout << " successfully create ssd detector" << std::endl;
            return std::unique_ptr<DetectorImpl>(new DetectorSSD());
    case DetetorType ::Retinanet:
        std::cout << " successfully create retinanet detector" << std::endl;
        return std::unique_ptr<DetectorImpl>(new DetectorRetinanet());
    case DetetorType::FasterRcnn:
        std::cout << " successfully create fasterrcnn detector" << std::endl;
        return std::unique_ptr<DetectorImpl>(new DetectorFasterRcnn());
    default:
            std::cout << "only support ssd, retinanet, faster_rcnn now" << std::endl;
    }
    return nullptr;
}
