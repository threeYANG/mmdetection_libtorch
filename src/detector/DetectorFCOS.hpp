#include "DetectorCommon.hpp"
#include "DetectorImpl.hpp"
#include "AnchorPointGenerator.hpp"

#define EPSILON 1e-6


class DetectorFCOS : public DetectorCommon , public DetectorImpl {
public:
    DetectorFCOS();
    virtual ~DetectorFCOS();
    void LoadParams(const Params& params, torch::DeviceType* device_type) override ;
    void LoadTracedModule() override;
    void Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) override ;

private:

    void get_anchors();

    void get_bboxes(const c10::IValue& output_data,
                    torch::Tensor& bboxes,
                    torch::Tensor& scores);

    AnchorPointGenerator point_generator_;

    std::vector<torch::Tensor> mlvl_points_;

};
