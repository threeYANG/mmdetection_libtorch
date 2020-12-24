#ifndef DETERCOTCOMMON_HPP
#define DETERCOTCOMMON_HPP

#include "torch/torch.h"
#include "torch/script.h"
#include "types.hpp"
#include "bbox.hpp"
#include "nms.hpp"
#include "transforms.hpp"
#include "AnchorPointGenerator.hpp"

using namespace std::chrono;

class DetectorCommon
{
public:
    DetectorCommon();
    ~DetectorCommon();

protected:

    void DetectOneStage(const cv::Mat& image,
                        std::vector<DetectedBox>& detected_boxes);

    void DetectTwoStage(const cv::Mat& image,
                        std::vector<DetectedBox>& detected_boxes);

    void LoadCommonParams(const Params& params,
                          torch::DeviceType* device_type);

    void LoadCommonTracedModule();

    c10::IValue backbone(const cv::Mat& image);

    virtual void first_stage(const cv::Mat& image,
                             torch::Tensor& bboxes,
                             torch::Tensor& scores);

    virtual void second_stage(const torch::Tensor& proposals,
                              torch::Tensor& bbox_results,
                              torch::Tensor& segm_results);

    virtual void get_bboxes(const c10::IValue& output,
                            torch::Tensor& bboxes,
                            torch::Tensor& scores);

    virtual void get_anchor_generators(const std::vector<int>& anchor_base_sizes,
                                const std::vector<float>& anchor_scales,
                                const std::vector<float>& anchor_ratios);

    //must know the net size
    virtual void get_anchors();


private:
    void init_params();

protected:

    torch::DeviceType * device_;
    DetetorType detector_type_;

    int net_width_;
    int net_height_;
    float conf_thresh_;
    std::vector<int> strides_;
    int nms_pre_;
    int use_sigmoid_;
    float nms_thresh_;
    float score_thresh_;
    int max_per_img_;

    TransformParams transform_params_;
    AnchorHeadParams anchor_head_params_;
    RPNHeadParams rpn_head_params_;
    FPNParams fpn_params_;

    std::vector<AnchorPointGenerator> anchor_generators_;
    torch::Tensor mlvl_anchors_;

    std::string backbone_module_path_;
    std::unique_ptr<torch::jit::script::Module> backbone_module_;

};

#endif // DETERCOTCOMMON_HPP
