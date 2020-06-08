#include "DetectorCommon.hpp"
#include "DetectorImpl.hpp"
#include "AnchorGenerator.hpp"

#define EPSILON 1e-6


class DetectorSSD : public DetectorCommon , public DetectorImpl {
public:
    DetectorSSD();
    virtual ~DetectorSSD();
    void LoadParams(const Params& params, torch::DeviceType* device_type) override ;
    void LoadTracedModule() override;
    void Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) override ;



private:
    void get_anchor_generators(const std::vector<int>& anchor_base_sizes,
                               const std::vector<float>& anchor_scales,
                               const std::vector<float>& anchor_ratios);
    //must know the net size
    void get_anchor_boxes();
    void get_min_max_size();

private:

    std::vector<std::vector<float>> ratios_;

    SSDHeadParams ssd_head_params_;

    std::vector<int>  min_sizes_;
    std::vector<int>  max_sizes_;
};
