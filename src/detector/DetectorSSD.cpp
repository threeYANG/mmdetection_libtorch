//
// Created by dl on 2020/1/3.
//

#include "DetectorSSD.hpp"

DetectorSSD::DetectorSSD() {
    detector_type_ = DetetorType ::SSD;
}

DetectorSSD::~DetectorSSD() {

}

void DetectorSSD::LoadParams(const Params& params, torch::DeviceType* device_type) {

    DetectorCommon::LoadParams(params, device_type);

    ssd_head_params_ = params.ssd_head_params_;

    //ssd is Fixed network input
    // anchor_boxes only calculat once
    //the input size is need to calculat min_max_size
    assert (transform_params_.keep_ratio_ == 0);
    assert (transform_params_.img_scale_[0] = transform_params_.img_scale_[1]);

    net_width_ = ssd_head_params_.input_size_;
    net_height_ = ssd_head_params_.input_size_;
    get_anchor_generators({},{},{});
}

void DetectorSSD::LoadTracedModule(){
    DetectorCommon::LoadTracedModule();
}

void DetectorSSD::get_min_max_size() {
    int min_ratio = int(ssd_head_params_.basesize_ratio_range_[0] * 100);
    int max_ratio = int(ssd_head_params_.basesize_ratio_range_[1] * 100);
    int step = int(floor(max_ratio - min_ratio) / float(ssd_head_params_.anchor_ratios_.size() - 2));
    int r = min_ratio;
    while(r < max_ratio + 1) {
        min_sizes_.push_back(int(net_width_ * r / 100));
        r += step;
        max_sizes_.push_back(int(net_width_ * r / 100));
    }
    if (net_width_ == 300) {
        if (abs(ssd_head_params_.basesize_ratio_range_[0]-0.15) <= EPSILON ) {
            min_sizes_.insert(std::begin(min_sizes_), int(net_width_ * 7 / 100));
            max_sizes_.insert(std::begin(max_sizes_), int(net_width_ * 15 / 100));
        } else if(abs(ssd_head_params_.basesize_ratio_range_[0]-0.2) <= EPSILON ) {
            min_sizes_.insert(std::begin(min_sizes_), int(net_width_ * 10 / 100));
            max_sizes_.insert(std::begin(max_sizes_), int(net_width_ * 20 / 100));
        } else {
            std::cout << "get_min_max_size failed 300" << std::endl;
        }

    } else if (net_width_ == 512) {
        if (abs(ssd_head_params_.basesize_ratio_range_[0]-0.1) <= EPSILON) {
            min_sizes_.insert(std::begin(min_sizes_), int(net_width_ * 4 / 100));
            max_sizes_.insert(std::begin(max_sizes_), int(net_width_ * 10 / 100));
        } else if (abs(ssd_head_params_.basesize_ratio_range_[0]-0.15) <= EPSILON) {
            min_sizes_.insert(std::begin(min_sizes_), int(net_width_ * 7 / 100));
            max_sizes_.insert(std::begin(max_sizes_), int(net_width_ * 15 / 100));
        } else {
            std::cout << "get_min_max_size failed  512" << std::endl;
        }
    }
}

void DetectorSSD::get_anchor_generators(const std::vector<int>& anchor_base_sizes,
                                        const std::vector<float>& anchor_scales,
                                        const std::vector<float>& anchor_ratios) {
    get_min_max_size();
    for (int k = 0; k < anchor_head_params_.anchor_strides_.size(); k++) {
         int base_size = min_sizes_[k];
         int stride = anchor_head_params_.anchor_strides_[k];
         std::vector<float> ctr(2);
         ctr[0] = ctr[1] = (stride - 1) / 2.0;
         std::vector<float> scales(2);
         scales[0] = 1.0;
         scales[1] = sqrt(float(max_sizes_[k]) / float(min_sizes_[k]));
         std::vector<float> ratios;
         ratios.push_back(1.0);
         // 4 or 6 ratio
         for (int i = 0; i < ssd_head_params_.anchor_ratios_[k].size(); i++) {
             ratios.push_back(1.0 / ssd_head_params_.anchor_ratios_[k][i]);
             ratios.push_back(ssd_head_params_.anchor_ratios_[k][i]);
         }
         ratios_.push_back(ratios);
         AnchorGenerator anchor_generator(base_size, scales, ratios, false, ctr);

         anchor_generator.feature_maps_sizes_.clear();
         anchor_generator.feature_maps_sizes_.push_back(ceil(float(net_height_) / stride));
         anchor_generator.feature_maps_sizes_.push_back(ceil(float(net_width_) / stride));

         anchor_generators_.push_back(anchor_generator);
    }
}

void DetectorSSD::get_anchor_boxes(){
   for (int k = 0; k < int(anchor_generators_.size()); k++) {

        std::vector<int> ind;
        for (int i = 0; i < ratios_[k].size(); i++)
            ind.push_back(i);
        ind.insert(ind.begin() + 1, int(ind.size()));
        torch::Tensor indices = torch::tensor(ind, torch::kLong);
        anchor_generators_[k].base_anchors_ = anchor_generators_[k].base_anchors_.index_select(0, indices);

        torch::Tensor all_anchors = anchor_generators_[k].grid_anchors(anchor_head_params_.anchor_strides_[k], *device_);
        if (k == 0) {
            mlvl_anchors_ = all_anchors;
        } else {
            mlvl_anchors_ = torch::cat({mlvl_anchors_, all_anchors}, 0);
        }
    }
}


void DetectorSSD::Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) {
    DetectorCommon::DetectOneStage(image, detected_boxes);
}







