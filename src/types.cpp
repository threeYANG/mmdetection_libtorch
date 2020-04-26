#include "types.hpp"
<<<<<<< HEAD
#include "utils/logging.hpp"
#include "utils/json/json.h"
#include "utils/json/json-forwards.h"
using namespace Json;
using namespace tunicornvideostruct;

std::vector<std::string> detector_types = {"SSD", "Retinanet"};


=======

#include "utils/logging.hpp"
#include "utils/json/json.h"
#include "utils/json/json-forwards.h"

std::vector<std::string> detector_types = {"SSD"};
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656

int find_type(std::string Detectedtype) {
    for(int i = 0; i < detector_types.size(); i++) {
        if(detector_types[i] == Detectedtype) {
            return i;
        }
    }
    return -1;
}

void print_type() {
    for(const auto& type: detector_types) {
        std::cout << type << " ";
    }
    std::cout << std::endl;
}

<<<<<<< HEAD
void ReadSSDParams(const Value& root, SSDParams& ssd_params){
    // in_channels
    const Value in_channels_arrays = root["SSD"]["in_channels"];
    for (const auto & in_channels_array : in_channels_arrays) {
        ssd_params.in_channels_.push_back(in_channels_array.asInt());
    }
    //basesize_ratio_range
    const Value max_size_arrays = root["SSD"]["basesize_ratio_range"];
    for (const auto & max_size_array : max_size_arrays) {
        ssd_params.basesize_ratio_range_.push_back(max_size_array.asFloat());
    }
    //aspect_ratios
    const Value aspect_ratios_arrays = root["SSD"]["anchor_ratios"];
    for (const auto & aspect_ratios_array : aspect_ratios_arrays) {
        std::vector<int> single;
        for(const auto & single_array: aspect_ratios_array) {
            single.push_back(single_array.asInt());
        }
        ssd_params.anchor_ratios_.push_back(single);
    }
}

void ReadRetinanetParams(const Value& root, RetinanetParams& Retinanet){
    // anchor_ratios
    const Value anchor_ratios_arrays = root["Retinanet"]["anchor_ratios"];
    for (const auto & anchor_ratios_array : anchor_ratios_arrays) {
        Retinanet.anchor_ratios_.push_back(anchor_ratios_array.asFloat());
    }
    Retinanet.in_channels_ = root["Retinanet"]["in_channels"].asInt();
    Retinanet.feat_channels_ = root["Retinanet"]["feat_channels"].asInt();
    Retinanet.stacked_convs_ = root["Retinanet"]["stacked_convs"].asInt();
    Retinanet.octave_base_scale_ = root["Retinanet"]["octave_base_scale"].asInt();
    Retinanet.scales_per_octave_ = root["Retinanet"]["scales_per_octave"].asInt();
}

int Params::Read(const std::string& config_file) {
=======
int Params::Read(const std::string& config_file) {
    using namespace Json;
    using namespace tunicornvideostruct;
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656
    Reader reader;
    Value root;
    std::ifstream ifs(config_file);
    bool ret = reader.parse(ifs, root);
    if(!ret) {
        std::cout<<"can not parse "<<config_file;
        return -1;
    }
<<<<<<< HEAD
    int type_ids = find_type(root["DetectorType"].asString());
    if (type_ids  == -1) {
=======

    int res = find_type(root["DetectorType"].asString());
    if (res == -1) {
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656
        std::cout << "only support:" <<std::endl;
        print_type();
        return -1;
    }
<<<<<<< HEAD
    detector_type_ = static_cast<DetetorType>(type_ids );
    model_path_ = root["modelPath"].asString();
    nms_thresh_ = root["nms_iou_thr"].asFloat();
    score_thresh_ = root["score_thr"].asFloat();
    conf_thresh_ = root["conf_thr"].asFloat();
    max_num_ = root["max_num"].asInt();
    nms_pre_ = root["nms_pre"].asInt();
    use_sigmoid_ = root["use_sigmoid"].asInt();
    //anchor_strides
    const Value step_arrays = root["anchor_strides"];
    for (const auto & step_array : step_arrays) {
        anchor_strides_.push_back(step_array.asInt());
    }
    //target_means_;
    const Value target_means_arrays = root["target_means"];
    for (const auto & target_means_array: target_means_arrays) {
        target_means_.push_back(target_means_array.asFloat());
    }
    //target_stds_;
    const Value target_stds_arrays = root["target_stds"];
    for (const auto & target_stds_array: target_stds_arrays) {
        target_stds_.push_back(target_stds_array.asFloat());
    }

    //mean_;
    const Value mean_arrays = root["Transform"]["mean"];
    for (const auto & mean_array: mean_arrays) {
        transform_params_.mean_.push_back(mean_array.asFloat());
    }
    //std_;
    const Value std_arrays = root["Transform"]["std"];
    for (const auto & std_array: std_arrays) {
        transform_params_.std_.push_back(std_array.asFloat());
    }
    //to_rgb
    transform_params_.to_rgb_ = root["Transform"]["to_rgb"].asInt();
    //img_scale_;
    const Value img_scale_arrays = root["Transform"]["img_scale"];
    for (const auto & img_scale_array: img_scale_arrays) {
        transform_params_.img_scale_.push_back(img_scale_array.asInt());
    }
    //keep_ratio_;
    transform_params_.keep_ratio_ = root["Transform"]["keep_ratio"].asInt();
    transform_params_.pad_ = root["Transform"]["pad"].asInt();

    if (detector_type_ == DetetorType::SSD ) {
       ReadSSDParams(root, ssd_params_);
    } else if(detector_type_ == DetetorType::Retinanet) {
       ReadRetinanetParams(root, retinanet_params_);
    }
    return 0;
}

=======
    detector_type_ = static_cast<DetetorType>(res);
    model_path_ = root["ModelPath"].asString();
    net_size_ = root["NetSize"].asInt();
    nms_thresh_ = root["nmsThresh"].asFloat();
    // SSD
    // feature_maps
    const Value feature_maps_arrays = root["SSD"]["FeatureMaps"];
    for (const auto & feature_maps_array : feature_maps_arrays) {
        ssd_params_.feature_maps_.push_back(feature_maps_array.asInt());
    }
    //steps
    const Value step_arrays = root["SSD"]["Steps"];
    for (const auto & step_array : step_arrays) {
        ssd_params_.steps_.push_back(step_array.asInt());
    }
    //min_size
    const Value min_size_arrays = root["SSD"]["MinSize"];
    for (const auto & min_size_array : min_size_arrays) {
        ssd_params_.min_size_.push_back(min_size_array.asInt());
    }
    //max_size
    const Value max_size_arrays = root["SSD"]["MaxSize"];
    for (const auto & max_size_array : max_size_arrays) {
        ssd_params_.max_size_.push_back(max_size_array.asInt());
    }
    //aspect_ratios
    const Value aspect_ratios_arrays = root["SSD"]["AspectRatios"];
    for (const auto & aspect_ratios_array : aspect_ratios_arrays) {
        std::vector<int> single;
        for(const auto & single_array: aspect_ratios_array) {
            single.push_back(single_array.asInt());
        }
        ssd_params_.aspect_ratios_.push_back(single);
    }
    return 0;
}
>>>>>>> b5eafd9d05aadaba42e84010fd92516eca896656
