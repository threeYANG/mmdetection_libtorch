#include "DetectorCommon.hpp"

DetectorCommon::DetectorCommon()
{
    module_ = nullptr;
    device_ = nullptr;
}

DetectorCommon::~DetectorCommon() {

}

void DetectorCommon::init_params() {
    net_width_ = 0;
    net_height_ = 0;
    conf_thresh_ = 0;
}

void DetectorCommon::LoadParams(const Params& params, torch::DeviceType* device_type) {

    init_params();
    device_ = device_type;
    model_path_ = params.model_path_;
    conf_thresh_ = params.conf_thresh_;

    transform_params_ = params.transform_params_;
    anchor_head_params_ = params.anchor_head_params_;
    rpn_head_params_ = params.rpn_head_params_;
    roi_extractor_params_ = params.roi_extractor_params_;
    fpn_params_ = params.fpn_params_;
}


void DetectorCommon::LoadTracedModule() {
    module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path_));
}

void DetectorCommon::get_anchor_generators(const std::vector<int>& anchor_base_sizes,
                            const std::vector<float>& anchor_scales,
                            const std::vector<float>& anchor_ratios) {
    std::vector<int> base_sizes;
    if (anchor_base_sizes.empty()) {
        base_sizes = anchor_head_params_.anchor_strides_;
    } else {
        base_sizes = anchor_base_sizes;
    }
    for (int k = 0; k < int(base_sizes.size()); k++) {
        AnchorGenerator anchor_generator(base_sizes[k], anchor_scales, anchor_ratios);
        anchor_generators_.push_back(anchor_generator);
    }

}
void DetectorCommon::get_anchor_boxes() {

    for (int k = 0; k < int(anchor_generators_.size()); k++) {
        anchor_generators_[k].feature_maps_sizes_.clear();
        anchor_generators_[k].feature_maps_sizes_.push_back(ceil(float(net_height_) / anchor_head_params_.anchor_strides_[k]));
        anchor_generators_[k].feature_maps_sizes_.push_back(ceil(float(net_width_) / anchor_head_params_.anchor_strides_[k]));
        torch::Tensor all_anchors = anchor_generators_[k].grid_anchors(anchor_head_params_.anchor_strides_[k], *device_);
        if (k == 0) {
            mlvl_anchors_ = all_anchors;
        } else {
            mlvl_anchors_ = torch::cat({mlvl_anchors_, all_anchors}, 0);
        }
    }

}


void DetectorCommon::get_proposals(torch::Tensor& output, const std::vector<int>& img_shape,
                                   torch::Tensor& proposals_bboxes, torch::Tensor&  proposals_scores,
                                   bool rpn) {

    assert(anchor_generators_.size() > 0);

    int digit_nums = output.size(1);

    if (anchor_head_params_.use_sigmoid_ == 1) {
        output.slice(1, 4, digit_nums).sigmoid_();
    } else {
        output.slice(1, 4, digit_nums) = output.slice(1, 4, digit_nums).softmax(1);
    }

    int start = 0;
    int end = 0;
    for (int k = 0; k < int(anchor_generators_.size()); k++) {
        int anchor_num = anchor_generators_[k].anchor_nums_;
        end = start + anchor_num;
        torch::Tensor anchors_layer = mlvl_anchors_.slice(0, start, end);
        torch::Tensor output_layer =output.slice(0, start, end);

        torch::Tensor score_layer = output_layer.slice(1, 4, digit_nums);
        torch::Tensor loc = output_layer.slice(1, 0, 4);
        if (anchor_head_params_.nms_pre_ > 0 && anchor_num > anchor_head_params_.nms_pre_){
            if (anchor_head_params_.use_sigmoid_ == 0){
                score_layer = score_layer.slice(1, 1, digit_nums);
            }
            std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(score_layer, 1);
            torch::Tensor max_scores= std::get<0>(max_classes);
            std::tuple<torch::Tensor, torch::Tensor> topk = torch::topk(max_scores, anchor_head_params_.nms_pre_, 0);
            torch::Tensor topk_inds = std::get<1>(topk);

            torch::Tensor max_output_layer = output_layer.index_select(0, topk_inds);
            anchors_layer = anchors_layer.index_select(0, topk_inds);
            loc = max_output_layer.slice(1, 0, 4);
            score_layer = max_output_layer.slice(1, 4, digit_nums);
        }
        torch::Tensor proposals_bboxes_layer= delta2bbox(anchors_layer, loc, img_shape, anchor_head_params_.target_means_, anchor_head_params_.target_stds_);
        if (rpn) {
            torch::Tensor inds = singleclass_nms(torch::cat({proposals_bboxes_layer, score_layer}, 1), anchor_head_params_.nms_thresh_);
            int num = cv::min(int(inds.sizes()[0]) , rpn_head_params_.nms_post_);
            inds = inds.slice(0, 0, num);
            proposals_bboxes_layer = proposals_bboxes_layer.index_select(0, inds);
            score_layer = score_layer.index_select(0, inds);
        }

        if( k == 0) {
            proposals_bboxes = proposals_bboxes_layer;
            proposals_scores = score_layer;
        } else {
            proposals_bboxes = torch::cat({proposals_bboxes, proposals_bboxes_layer}, 0);
            proposals_scores = torch::cat({proposals_scores, score_layer}, 0);
        }
        start = end;
    }
    assert(proposals_bboxes.sizes()[0] == proposals_scores.sizes()[0]);
}

void DetectorCommon::DetectOneStage(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) {
    torch::Tensor tensor_image;
    transform(image, tensor_image, transform_params_, net_width_, net_height_, device_);
    std::cout << "tensor_image:"<<std::endl;
    std::cout << (tensor_image.squeeze())[0][100][100] <<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor output = module_->forward({tensor_image}).toTensor().squeeze(0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "forward taken : " << duration.count() << " ms" << std::endl;
    std::cout << output.sizes() << std::endl;


    get_anchor_boxes();
    torch::Tensor  proposals_scores;
    torch::Tensor  proposals_bboxes;
    bool rpn = false;
    get_proposals(output, transform_params_.img_shape_, proposals_bboxes, proposals_scores,  rpn);

    if (anchor_head_params_.use_sigmoid_ == 1) {
        torch::Tensor padding = torch::zeros({proposals_scores.sizes()[0], 1}).to(proposals_scores.device());
        proposals_scores = torch::cat({padding, proposals_scores}, 1);
    }

    torch::Tensor nms_results = multiclass_nms(proposals_bboxes, proposals_scores, anchor_head_params_.score_thresh_,
                                               anchor_head_params_.nms_thresh_,anchor_head_params_.max_per_img_);

    cv::Size2f scale = cv::Size2f(image.size().width / float(net_width_), image.size().height / float(net_height_));
    bbox2result(nms_results, conf_thresh_, scale, detected_boxes);
}

void DetectorCommon::get_rpn_fpn_data(const torch::Tensor& output, torch::Tensor& rpn_data, std::vector<torch::Tensor>& fpn_datas) {
    int rpn_num = 0;
    int fpn_num = 0;
    int rpn_channels = 4 + rpn_head_params_.class_num_ -1;
    int fpn_channels = fpn_params_.out_channels_;
    for(int k = 0; k < int(anchor_generators_.size()); k++) {
        rpn_num += anchor_generators_[k].anchor_nums_ * rpn_channels;
        assert (anchor_generators_[k].feature_maps_sizes_.size() == 2);
        fpn_num += anchor_generators_[k].feature_maps_sizes_[0] * anchor_generators_[k].feature_maps_sizes_[1] * fpn_channels;
    }
    rpn_data = output.slice(0, 0, rpn_num).view({-1, rpn_channels});
    std::cout << rpn_data.sizes() << std::endl;
    std::cout << rpn_data[100] <<  std::endl;
    torch::Tensor fpn_data = output.slice(0, rpn_num, output.sizes()[0]);
    int start = 0;
    int end = 0;
    for(int k = 0; k < int(anchor_generators_.size()); k++) {
      int feature_height = anchor_generators_[k].feature_maps_sizes_[0];
      int feature_width = anchor_generators_[k].feature_maps_sizes_[1];
      start = end;
      int fpn_num_layer = feature_height * feature_width * fpn_channels;
      end = end + fpn_num_layer;
      torch::Tensor fpn_data_layer = fpn_data.slice(0, start, end).view({fpn_channels, feature_height, feature_width});
      std::cout << fpn_data_layer.sizes() << std::endl;
      std::cout << fpn_data_layer[100][10][10]<<std::endl;
      fpn_datas.push_back(fpn_data_layer);
    }
}


void DetectorCommon::DetectTwoStage(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) {
        torch::Tensor tensor_image;
        transform(image, tensor_image, transform_params_, net_width_, net_height_, device_);

        std::cout << "tensor_image.sizes(): "<<std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor output = module_->forward({tensor_image}).toTensor().squeeze(0);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        std::cout << "forward taken : " << duration.count() << " ms" << std::endl;
        std::cout << output.sizes() << std::endl;


        get_anchor_boxes();

        torch::Tensor rpn_data;
        std::vector<torch::Tensor> fpn_datas;
        get_rpn_fpn_data(output, rpn_data, fpn_datas);


        torch::Tensor  proposals_scores;
        torch::Tensor  proposals_bboxes;
        bool rpn = true;
        get_proposals(rpn_data, transform_params_.img_shape_, proposals_bboxes, proposals_scores,  rpn);

        torch::Tensor proposals = torch::cat({proposals_bboxes, proposals_scores}, 1);
        if (rpn_head_params_.nms_across_levels_ == 1) {

            torch::Tensor inds = singleclass_nms(proposals, anchor_head_params_.nms_thresh_);
            int num = cv::min(int(inds.sizes()[0]), rpn_head_params_.max_num_);
            proposals = proposals.slice(0, 0, num);
        } else {
            int num = cv::min(rpn_head_params_.max_num_, int(proposals_scores.sizes()[0]));
            std::tuple<torch::Tensor, torch::Tensor> topk = torch::topk(proposals_scores.squeeze(), num);
            torch::Tensor topk_inds = std::get<1>(topk);
            proposals = proposals.index_select(0, topk_inds);
        }

        std::cout << proposals.sizes() << std::endl;
        std::cout << proposals[0] << std::endl;
        std::cout << proposals[1] << std::endl;
        std::cout << proposals[2] << std::endl;
        std::cout << proposals[3] << std::endl;
        std::cout << proposals[4] << std::endl;

        torch::Tensor rois = bbox2roi(proposals);

        std::cout << rois[0]  << std::endl;
}




