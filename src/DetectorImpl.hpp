//
// Created by dl on 2020/1/2.
//

#ifndef DETECTOR_DETECTORIMPL_HPP
#define DETECTOR_DETECTORIMPL_HPP
#include <algorithm>
#include "torch/torch.h"
#include "torch/script.h"
#include "types.hpp"



class DetectorImpl {
public:
    DetectorImpl();
    virtual  ~DetectorImpl();

    static  std::unique_ptr<DetectorImpl> Create(DetetorType detetorType);

    virtual void LoadParams(const Params& params, torch::DeviceType* device_type) = 0;
    virtual void LoadTracedModule() = 0;
    virtual void Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) = 0;
};

class DetectorCreator final
{
public:
    DetectorCreator() = default;
    ~DetectorCreator() = default;

    static std::unique_ptr<DetectorImpl> create_detector(DetetorType detector_type);
};



#endif //DETECTOR_DETECTORIMPL_HPP
