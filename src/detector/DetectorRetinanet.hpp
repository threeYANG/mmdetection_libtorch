#ifndef DETECTORRETINANET_HPP
#define DETECTORRETINANET_HPP

#include "SingleStage.hpp"
#include "DetectorImpl.hpp"
#include "AnchorGenerator.hpp"

class DetectorRetinanet: public SingleStage , public DetectorImpl {

public:
    DetectorRetinanet();
    virtual ~DetectorRetinanet();

    void LoadParams(const Params& params, torch::DeviceType* device_type) override ;
    void LoadTracedModule() override;
    void Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) override ;

private:
    int in_channels_;
    int stacked_convs_;
    int feat_channels_;
    int octave_base_scale_;
    int scales_per_octave_;
    std::vector<float> anchor_ratios_;


};

#endif // DETECTORRETINANET_HPP
