#ifndef DETERCOTCOMMON_HPP
#define DETERCOTCOMMON_HPP

#include "torch/torch.h"
#include "torch/script.h"
#include "types.hpp"
#include "bbox.hpp"
#include "transforms.hpp"
#include "AnchorGenerator.hpp"

using namespace std::chrono;

class DetectorCommon
{
public:
    DetectorCommon();
    ~DetectorCommon();

    void LoadParams(const Params& params, torch::DeviceType* device_type);
    void LoadTracedModule();
    void DetectOneStage(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes);
    void DetectTwoStage(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes);

    virtual void get_anchor_generators(const std::vector<int>& anchor_base_sizes,
                                const std::vector<float>& anchor_scales,
                                const std::vector<float>& anchor_ratios);
    //must know the net size
    virtual void get_anchor_boxes();

private:
    void init_params();
    void get_rpn_fpn_data(const torch::Tensor& output, torch::Tensor& rpn_data, std::vector<torch::Tensor>& fpn_datas);
    void get_proposals(torch::Tensor& output, const std::vector<int>& img_shape,
                       torch::Tensor& proposals_bboxes, torch::Tensor&  proposals_scores,
                       bool rpn);


protected:

    torch::DeviceType * device_;

    DetetorType detector_type_;
    std::string model_path_;
    int net_width_;
    int net_height_;
    float conf_thresh_;

    TransformParams transform_params_;
    AnchorHeadParams anchor_head_params_;
    RPNHeadParams rpn_head_params_;
    RoiExtractorParams roi_extractor_params_;
    FPNParams fpn_params_;

    std::vector<AnchorGenerator> anchor_generators_;
    torch::Tensor mlvl_anchors_;
    std::unique_ptr<torch::jit::script::Module> module_;
};

#endif // DETERCOTCOMMON_HPP
