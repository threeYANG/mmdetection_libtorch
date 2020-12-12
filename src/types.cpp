#include "types.hpp"
#include <algorithm>
#include "utils/logging.hpp"
#include "utils/json/json.h"
#include "utils/json/json-forwards.h"
using namespace Json;
using namespace tunicornvideostruct;

std::vector<std::string> detector_types = {"SSD", "Retinanet", "FasterRCNN", "FCOS"};



int find_type(std::string Detectedtype) {
    std::transform(Detectedtype.begin(), Detectedtype.end(), Detectedtype.begin(), toupper);
    for(int i = 0; i < detector_types.size(); i++) {
        std::transform(detector_types[i].begin(), detector_types[i].end(), detector_types[i].begin(), toupper);
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


void ReadTransformParams(const Value& root, TransformParams& transform) {
    //mean_;
    const Value mean_arrays = root["Transforms"]["mean"];
    for (const auto & mean_array: mean_arrays) {
        transform.mean_.push_back(mean_array.asFloat());
    }
    //std_;
    const Value std_arrays = root["Transforms"]["std"];
    for (const auto & std_array: std_arrays) {
        transform.std_.push_back(std_array.asFloat());
    }
    //to_rgb
    transform.to_rgb_ = root["Transforms"]["to_rgb"].asInt();
    //img_scale_;
    const Value img_scale_arrays = root["Transforms"]["img_scale"];
    for (const auto & img_scale_array: img_scale_arrays) {
        transform.img_scale_.push_back(img_scale_array.asInt());
    }
    //keep_ratio_;
    transform.keep_ratio_ = root["Transforms"]["keep_ratio"].asInt();
    transform.pad_ = root["Transforms"]["pad"].asInt();
}

void ReadAnchorHeadParams(const Value& root, AnchorHeadParams& anchor_head) {
    //target_means_;
    const Value target_means_arrays = root["AnchorHead" ]["target_means"];
    for (const auto & target_means_array: target_means_arrays) {
        anchor_head.target_means_.push_back(target_means_array.asFloat());
    }
    //target_stds_;
    const Value target_stds_arrays = root["AnchorHead" ]["target_stds"];
    for (const auto & target_stds_array: target_stds_arrays) {
        anchor_head.target_stds_.push_back(target_stds_array.asFloat());
    }
}

void ReadRetinaHeadParams(const Value& root, RetinaHeadParams& RetinaHead) {
    // anchor_ratios
    const Value anchor_ratios_arrays = root["RetinaHead"]["anchor_ratios"];
    for (const auto & anchor_ratios_array : anchor_ratios_arrays) {
        RetinaHead.anchor_ratios_.push_back(anchor_ratios_array.asFloat());
    }
    RetinaHead.octave_base_scale_ = root["RetinaHead"]["octave_base_scale"].asInt();
    RetinaHead.scales_per_octave_ = root["RetinaHead"]["scales_per_octave"].asInt();
}

void ReadSSDHeadParams(const Value& root, SSDHeadParams& SSDHead) {
    //basesize_ratio_range
    SSDHead.input_size_ = root["SSDHead"]["input_size"].asInt();
    const Value max_size_arrays = root["SSDHead"]["basesize_ratio_range"];
    for (const auto & max_size_array : max_size_arrays) {
        SSDHead.basesize_ratio_range_.push_back(max_size_array.asFloat());
    }
    //aspect_ratios
    const Value aspect_ratios_arrays = root["SSDHead"]["anchor_ratios"];
    for (const auto & aspect_ratios_array : aspect_ratios_arrays) {
        std::vector<int> single;
        for(const auto & single_array: aspect_ratios_array) {
            single.push_back(single_array.asInt());
        }
        SSDHead.anchor_ratios_.push_back(single);
    }
}

void ReadRPNHeadParams(const Value& root, RPNHeadParams& RPNHead) {
    const Value anchor_ratios_arrays = root["RPNHead"]["anchor_ratios"];
    for (const auto & anchor_ratios_array : anchor_ratios_arrays) {
        RPNHead.anchor_ratios_.push_back(anchor_ratios_array.asFloat());
    }
    const Value anchor_scales_arrays = root["RPNHead"]["anchor_scales"];
    for (const auto & anchor_scales_array : anchor_scales_arrays) {
        RPNHead.anchor_scales_.push_back(anchor_scales_array.asFloat());
    }

     RPNHead.nms_across_levels_ = root["RPNHead"]["nms_across_levels"].asInt();
     RPNHead.nms_post_ = root["RPNHead"]["nms_post"].asInt();
     RPNHead.max_num_ = root["RPNHead"]["max_num"].asInt();
     RPNHead.min_bbox_size_ = root["RPNHead"]["min_bbox_size"].asInt();
     RPNHead.class_num_ = root["RPNHead"]["class_num"].asInt();

}

void ReadRoiExtractorParams(const Value& root, RoiExtractorParams& RoiExtractor) {
    RoiExtractor.type_ = root["RoiExtractor"]["type"].asString();
    RoiExtractor.out_size_ = root["RoiExtractor"]["out_size"].asInt();
    RoiExtractor.sampling_ratio_ = root["RoiExtractor"]["sampling_ratio"].asInt();
    RoiExtractor.out_channels_ = root["RoiExtractor"]["out_channels"].asInt();

    const Value featmap_strides_arrays = root["RoiExtractor"]["featmap_strides"];
    for (const auto & featmap_strides_array : featmap_strides_arrays) {
        RoiExtractor.featmap_strides_.push_back(featmap_strides_array.asInt());
    }
    //target_means_;
    const Value target_means_arrays = root["RoiExtractor" ]["target_means"];
    for (const auto & target_means_array: target_means_arrays) {
        RoiExtractor.target_means_.push_back(target_means_array.asFloat());
    }
    //target_stds_;
    const Value target_stds_arrays = root["RoiExtractor" ]["target_stds"];
    for (const auto & target_stds_array: target_stds_arrays) {
        RoiExtractor.target_stds_.push_back(target_stds_array.asFloat());
    }
}

void ReadFPNParams(const Value& root, FPNParams& fpn_params) {
    fpn_params.out_channels_ = root["FPN"]["out_channels"].asInt();
    fpn_params.num_outs_ = root["FPN"]["num_outs"].asInt();
}

int Params::Read(const std::string& config_file) {
    Reader reader;
    Value root;
    std::ifstream ifs(config_file);
    bool ret = reader.parse(ifs, root);
    if(!ret) {
        std::cout<<"can not parse "<<config_file;
        return -1;
    }
    int type_ids = find_type(root["DetectorType"].asString());
    if (type_ids  == -1) {
        std::cout << "only support:" <<std::endl;
        print_type();
        return -1;
    }
    detector_type_ = static_cast<DetetorType>(type_ids );
    module_path_ = root["modelPath"].asString();
    conf_thresh_ = root["conf_thr"].asFloat();
    //strides
    const Value step_arrays = root["strides"];
    for (const auto & step_array : step_arrays) {
        strides_.push_back(step_array.asInt());
    }

    //nms_pre_;
    nms_pre_ = root["nms_pre"].asInt();
    //use_sigmoid_;
    use_sigmoid_ = root["use_sigmoid"].asInt();
    //nms_thresh_;
    nms_thresh_ = root["nms_iou_thr"].asFloat();
    //max_per_img_
    max_per_img_ = root["max_per_img"].asInt();
    //score_thresh_
    score_thresh_ = root["score_thr"].asFloat();

    ReadTransformParams(root, transform_params_);


    if (detector_type_ == DetetorType::SSD ) {
        ReadAnchorHeadParams(root, anchor_head_params_);
        ReadSSDHeadParams(root, ssd_head_params_);
    } else if(detector_type_ == DetetorType::Retinanet) {
        ReadAnchorHeadParams(root, anchor_head_params_);
        ReadRetinaHeadParams(root, retina_head_params_);
    } else if (detector_type_ == DetetorType::FasterRcnn) {
        ReadAnchorHeadParams(root, anchor_head_params_);
        ReadRPNHeadParams(root, rpn_head_params_);
        ReadRoiExtractorParams(root, roi_extractor_params_);
        ReadFPNParams(root, fpn_params_);
    }
    return 0;
}

