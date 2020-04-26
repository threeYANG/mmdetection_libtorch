//
// Created by dl on 2020/1/2.
//

#ifndef DETECTOR_TYPES_HPP
#define DETECTOR_TYPES_HPP
#include <vector>
#include <string>
#include<opencv2/opencv.hpp>


<<<<<<< HEAD

=======
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656
struct DetectedBox {
    cv::Rect box;
    int label;
    float score;

    DetectedBox()
    {
        score = 0;
        label = -1;
    }
};


enum class DetetorType : int
{
<<<<<<< HEAD
    SSD = 0,
    Retinanet = 1,
=======
    SSD = 0
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656
};


struct SSDParams {
<<<<<<< HEAD
    std::vector<int> in_channels_;
    std::vector<float> basesize_ratio_range_;
    std::vector<std::vector<int>> anchor_ratios_;
};

struct RetinanetParams {
    int in_channels_;
    int stacked_convs_;
    int feat_channels_;
    int octave_base_scale_;
    int scales_per_octave_;
    std::vector<float> anchor_ratios_;
};

struct TransformParams {
    std::vector<float> mean_;
    std::vector<float> std_;
    int to_rgb_;
    std::vector<int> img_scale_;
    int keep_ratio_;
    int pad_;
=======
    std::vector<int> feature_maps_;
    std::vector<int> steps_;
    std::vector<int> min_size_;
    std::vector<int> max_size_;
    std::vector<std::vector<int>> aspect_ratios_;
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656
};

struct Params {
public:
<<<<<<< HEAD
    Params() = default;
    ~Params() = default;

    DetetorType detector_type_;
    std::string model_path_;
    float nms_thresh_;
    float score_thresh_;
    int max_num_;
    float conf_thresh_;

    int nms_pre_;
    int use_sigmoid_;
    //bbox head
    std::vector<int> anchor_strides_;
    std::vector<float> target_means_;
    std::vector<float> target_stds_;

    TransformParams transform_params_;

    //ssd 参数
    SSDParams ssd_params_;
    //Retinanet 参数
    RetinanetParams retinanet_params_;
=======
    Params()=default;
    ~Params()=default;

    DetetorType detector_type_;
    std::string model_path_;
    int net_size_;
    float nms_thresh_;

    //ssd 参数
    SSDParams ssd_params_;
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656

    int Read(const std::string &config_file);
};
#endif //DETECTOR_TYPES_HPP
