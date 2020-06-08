#ifndef DETECTORFASTERRCNN_HPP
#define DETECTORFASTERRCNN_HPP

#include "DetectorImpl.hpp"
#include "DetectorCommon.hpp"

class DetectorFasterRcnn: public DetectorCommon , public DetectorImpl
{
public:
    DetectorFasterRcnn();

    virtual ~DetectorFasterRcnn();

    void LoadParams(const Params& params, torch::DeviceType* device_type) override ;
    void LoadTracedModule() override;
    void Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) override ;

};

#endif // DETECTORFASTERRCNN_HPP
