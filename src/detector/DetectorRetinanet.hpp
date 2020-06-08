#ifndef DETECTORRETINANET_HPP
#define DETECTORRETINANET_HPP

#include "DetectorCommon.hpp"
#include "DetectorImpl.hpp"

class DetectorRetinanet: public DetectorCommon , public DetectorImpl {

public:
    DetectorRetinanet();
    virtual ~DetectorRetinanet();

    void LoadParams(const Params& params, torch::DeviceType* device_type) override ;
    void LoadTracedModule() override;
    void Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) override ;

private:

    RetinaHeadParams retina_head_params_;

};

#endif // DETECTORRETINANET_HPP
