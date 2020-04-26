//
// Created by threeYANG on 2020/1/2.
//

#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include "types.hpp"
#include<memory>
#include "torch/torch.h"


class DetectorImpl;

class Detector {
public:
    Detector();
    ~Detector();

<<<<<<< HEAD
    long Create(DetetorType detetorType);
=======
    void Create(DetetorType detetorType);
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656

    void LoadParams(const Params& params, torch::DeviceType* device_type);

    void LoadTracedModule();

    long Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes);

private:
    std::unique_ptr<DetectorImpl> impl_;
};


#endif //DETECTOR_HPP
