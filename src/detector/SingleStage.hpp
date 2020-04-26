#ifndef SINGLESTAGE_HPP
#define SINGLESTAGE_HPP

#include "torch/torch.h"
#include "torch/script.h"
#include "types.hpp"
#include "bbox.hpp"
#include "transforms.hpp"
#include "AnchorGenerator.hpp"

using namespace std::chrono;

class SingleStage
{
public:
    SingleStage();
    ~SingleStage();

    void LoadParams_Single(const Params& params, torch::DeviceType* device_type);
    void LoadTracedModule_Single();
    void Detect_Single(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes);

    virtual void get_anchor_generators(const std::vector<int>& anchor_base_sizes,
                                const std::vector<float>& anchor_scales,
                                const std::vector<float>& anchor_ratios);
    //must know the net size
    virtual void get_anchor_boxes();
    void get_featuremaps_sizes();

private:
    void init_params();
    void get_maximum_scores(torch::Tensor& output, int digit_nums);


protected:

    std::vector<int> anchor_strides_;
    std::vector<float> target_means_;
    std::vector<float> target_stds_;
    torch::DeviceType * device_;

    DetetorType detector_type_;
    std::string model_path_;
    int net_width_;
    int net_height_;
    float nms_thresh_;
    float scores_thresh_;
    int max_num_;
    float conf_thresh_;
    int nms_pre_;
    int use_sigmoid_;

    TransformParams transform_params_;

    std::vector<std::vector<int>> feature_maps_;

    std::vector<AnchorGenerator> anchor_generators_;
    torch::Tensor mlvl_anchors_;

    std::unique_ptr<torch::jit::script::Module> module_;
};

#endif // SINGLESTAGE_HPP
