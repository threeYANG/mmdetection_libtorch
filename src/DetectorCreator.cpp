//
// Created by dl on 2020/1/3.
//

#include "DetectorImpl.hpp"
#include "DetectorSSD.hpp"
<<<<<<< HEAD
#include "DetectorRetinanet.hpp"
=======
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656

std::unique_ptr<DetectorImpl> DetectorCreator::create_detector(DetetorType detector_type) {
    switch(detector_type) {
        case DetetorType ::SSD:
            std::cout << " successfully create ssd detector" << std::endl;
            return std::unique_ptr<DetectorImpl>(new DetectorSSD());
<<<<<<< HEAD
    case DetetorType ::Retinanet:
        std::cout << " successfully create retinanet detector" << std::endl;
        return std::unique_ptr<DetectorImpl>(new DetectorRetinanet());
    default:
            std::cout << "only support ssd, retinanet now" << std::endl;
    }
    return nullptr;
}
=======
        default:
            std::cout << "only support ssd now" << std::endl;
    }
    return nullptr;
}
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656
