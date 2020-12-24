#include "DetectorCommon.hpp"

void split(const std::string& s, std::vector<std::string>& tokens, char delim = ' ') {
    tokens.clear();
    auto string_find_first_not = [s, delim](size_t pos = 0) -> size_t {
        for (size_t i = pos; i < s.size(); i++) {
            if (s[i] != delim) return i;
        }
        return std::string::npos;
    };
    size_t lastPos = string_find_first_not(0);
    size_t pos = s.find(delim, lastPos);
    while (lastPos != std::string::npos) {
        tokens.emplace_back(s.substr(lastPos, pos - lastPos));
        lastPos = string_find_first_not(pos);
        pos = s.find(delim, lastPos);
    }
}

DetectorCommon::DetectorCommon()
{
    backbone_module_ = nullptr;
    device_ = nullptr;
}

DetectorCommon::~DetectorCommon() {

}

void DetectorCommon::init_params() {
    net_width_ = 0;
    net_height_ = 0;
    conf_thresh_ = 0;
}

void DetectorCommon::LoadCommonParams(const Params& params, torch::DeviceType* device_type) {

    init_params();
    device_ = device_type;
    conf_thresh_ = params.conf_thresh_;
    strides_ = params.strides_;
    nms_pre_ = params.nms_pre_ ;
    use_sigmoid_ = params.use_sigmoid_ ;
    nms_thresh_= params.nms_thresh_ ;
    score_thresh_ = params.score_thresh_ ;
    max_per_img_ = params.max_per_img_ ;

    transform_params_ = params.transform_params_;
    anchor_head_params_ = params.anchor_head_params_;
    rpn_head_params_ = params.rpn_head_params_;
    fpn_params_ = params.fpn_params_;
    backbone_module_path_ = params.module_path_;
}


void DetectorCommon::LoadCommonTracedModule() {

    backbone_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(backbone_module_path_));
}

void DetectorCommon::get_anchor_generators(const std::vector<int>& anchor_base_sizes,
                            const std::vector<float>& anchor_scales,
                            const std::vector<float>& anchor_ratios) {
    std::vector<int> base_sizes;
    if (anchor_base_sizes.empty()) {
        base_sizes = strides_;
    } else {
        base_sizes = anchor_base_sizes;
    }
    for (int k = 0; k < int(base_sizes.size()); k++) {
        AnchorPointGenerator anchor_generator(base_sizes[k], anchor_scales, anchor_ratios);
        anchor_generators_.push_back(anchor_generator);
    }

}
void DetectorCommon::get_anchors() {

    for (int k = 0; k < int(strides_.size()); k++) {
        anchor_generators_[k].feature_maps_sizes_.clear();
        anchor_generators_[k].feature_maps_sizes_.push_back(ceil(float(net_height_) / strides_[k]));
        anchor_generators_[k].feature_maps_sizes_.push_back(ceil(float(net_width_) / strides_[k]));
        torch::Tensor all_anchors = anchor_generators_[k].grid_anchors(strides_[k], *device_);
        if (k == 0) {
            mlvl_anchors_ = all_anchors;
        } else {
            mlvl_anchors_ = torch::cat({mlvl_anchors_, all_anchors}, 0);
        }
    }

}


void DetectorCommon::get_bboxes(const c10::IValue& output_data,
                               torch::Tensor& bboxes,
                               torch::Tensor&  scores){

    torch::Tensor cls_score = output_data.toTuple()->elements()[0].toTensor();
    torch::Tensor bbox_pred = output_data.toTuple()->elements()[1].toTensor();

    assert(anchor_generators_.size() > 0);

    if (use_sigmoid_ == 1) {
        cls_score.sigmoid_();
    }

    int start = 0;
    int end = 0;

    for (int k = 0; k < int(strides_.size()); k++) {
        int anchor_num = anchor_generators_[k].anchor_nums_;
        end = start + anchor_num;

        torch::Tensor anchors_layer = mlvl_anchors_.slice(0, start, end);
        torch::Tensor score_layer = cls_score.slice(0, start, end);
        torch::Tensor bboxes_pred_layer = bbox_pred.slice(0, start, end);

        if (nms_pre_ > 0 && anchor_num > nms_pre_){
            std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(score_layer, 1);
            torch::Tensor max_scores= std::get<0>(max_classes);
            std::tuple<torch::Tensor, torch::Tensor> topk = torch::topk(max_scores, nms_pre_, 0);
            torch::Tensor topk_inds = std::get<1>(topk);

            anchors_layer = anchors_layer.index_select(0, topk_inds);
            bboxes_pred_layer = bboxes_pred_layer.index_select(0, topk_inds);
            score_layer = score_layer.index_select(0, topk_inds);
        }

        torch::Tensor bboxes_layer= delta2bbox(anchors_layer, bboxes_pred_layer, transform_params_.img_shape_, anchor_head_params_.target_means_, anchor_head_params_.target_stds_);
        if( k == 0) {
            bboxes  = bboxes_layer;
            scores = score_layer;

        } else {
            bboxes = torch::cat({bboxes, bboxes_layer}, 0);
            scores = torch::cat({scores, score_layer}, 0);
        }
        start = end;
    }
    assert(bboxes.sizes()[0] == scores.sizes()[0]);
}

c10::IValue DetectorCommon::backbone(const cv::Mat& image) {

    torch::Tensor tensor_image;
    transform(image, tensor_image, transform_params_, net_width_, net_height_, device_);
    auto start = std::chrono::high_resolution_clock::now();
    c10::IValue output = backbone_module_->forward({tensor_image});
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "forward taken : " << duration.count() << " ms" << std::endl;
    return output;
}


void DetectorCommon::second_stage(const torch::Tensor& proposals,
                                  torch::Tensor& bbox_results,
                                  torch::Tensor& segm_results) {

}

void DetectorCommon::first_stage(const cv::Mat& image,
                                 torch::Tensor& bboxes,
                                 torch::Tensor& scores) {
    // get data stream from backone
    auto output = backbone(image);
    // get anchor boxes
    get_anchors();
    // get bboxes and scores from data stream and anchor boxes
    get_bboxes(output, bboxes, scores);
}
void DetectorCommon::DetectOneStage(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) {
    torch::Tensor  scores;
    torch::Tensor  bboxes;
    first_stage(image, bboxes, scores);

    if (use_sigmoid_ == 1) {
        torch::Tensor padding = torch::zeros({scores.sizes()[0], 1}).to(scores.device());
        scores = torch::cat({scores, padding}, 1);
    }

    torch::Tensor scale_factor = torch::tensor(transform_params_.scale_factor_);
    bboxes = bboxes / scale_factor;

    torch::Tensor nms_results = multiclass_nms(bboxes, scores, score_thresh_,
                                               nms_thresh_, max_per_img_);

    torch::Tensor segm_result;
    bbox2result(nms_results, segm_result, conf_thresh_, detected_boxes);
}

void DetectorCommon::DetectTwoStage(const cv::Mat& image,
                    std::vector<DetectedBox>& detected_boxes) {

    torch::Tensor  proposals_scores;
    torch::Tensor  proposals_bboxes;
    first_stage(image, proposals_bboxes, proposals_scores);

    torch::Tensor proposals = torch::cat({proposals_bboxes, proposals_scores.unsqueeze_(1)}, 1);

    torch::Tensor bbox_results;
    torch::Tensor segm_results;
    second_stage(proposals, bbox_results, segm_results);

    bbox2result(bbox_results, segm_results, conf_thresh_, detected_boxes);

}






