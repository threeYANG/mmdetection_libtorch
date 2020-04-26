//
// Created by dl on 2020/1/2.
//

#ifndef DETECTOR_DETECTORIMPL_HPP
#define DETECTOR_DETECTORIMPL_HPP
<<<<<<< HEAD
#include <algorithm>
#include "torch/torch.h"
#include "torch/script.h"
#include "types.hpp"

=======
#include "torch/torch.h"
#include "torch/script.h"
#include "types.hpp"
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656

using namespace std::chrono;

class DetectorImpl {
public:
    DetectorImpl();
    virtual  ~DetectorImpl();
<<<<<<< HEAD

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
=======

    static  std::unique_ptr<DetectorImpl> Create(DetetorType detetorType);

    virtual void LoadParams(const Params& params, torch::DeviceType* device_type) = 0;
    virtual void LoadTracedModule() = 0;
    virtual void Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) = 0;
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656
};

class DetectorCreator final
{
public:
    DetectorCreator() = default;
    ~DetectorCreator() = default;

    static std::unique_ptr<DetectorImpl> create_detector(DetetorType detector_type);
};




#endif //DETECTOR_DETECTORIMPL_HPP
