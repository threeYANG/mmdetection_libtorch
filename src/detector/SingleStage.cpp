#include "SingleStage.hpp"

SingleStage::SingleStage()
{
    module_ = nullptr;
    device_ = nullptr;
}

SingleStage::~SingleStage() {

}

void SingleStage::init_params() {

    net_width_ = 0;
    net_height_ = 0;
    nms_thresh_ = 0;
    scores_thresh_ = 0;
    conf_thresh_ = 0;
    max_num_ = 0;
    nms_pre_ = -1;
    use_sigmoid_ = 0; // default softmax
}

void SingleStage::LoadParams_Single(const Params& params, torch::DeviceType* device_type) {

    init_params();
    device_ = device_type;
    model_path_ = params.model_path_;
    nms_thresh_ = params.nms_thresh_;
    scores_thresh_ = params.score_thresh_;
    anchor_strides_ = params.anchor_strides_;
    target_means_ = params.target_means_;
    target_stds_ = params.target_stds_;
    conf_thresh_ = params.conf_thresh_;
    max_num_ = params.max_num_;
    use_sigmoid_ = params.use_sigmoid_;
    nms_pre_ = params.nms_pre_;

    transform_params_ = params.transform_params_;
}


void SingleStage::LoadTracedModule_Single() {
    module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path_));
}

void SingleStage::get_anchor_generators(const std::vector<int>& anchor_base_sizes,
                            const std::vector<float>& anchor_scales,
                            const std::vector<float>& anchor_ratios) {
    std::vector<int> base_sizes;
    if (anchor_base_sizes.empty()) {
        base_sizes = anchor_strides_;
    } else {
        base_sizes = anchor_base_sizes;
    }
    for (int k = 0; k < int(base_sizes.size()); k++) {
        AnchorGenerator anchor_generator(base_sizes[k], anchor_scales, anchor_ratios);
        anchor_generators_.push_back(anchor_generator);
    }

}
void SingleStage::get_anchor_boxes() {
    assert(anchor_generators_.size() == feature_maps_.size());
    for (int k = 0; k < int(anchor_generators_.size()); k++) {
        torch::Tensor all_anchors = anchor_generators_[k].grid_anchors({feature_maps_[k][0], feature_maps_[k][1]}, anchor_strides_[k], *device_);
        if (k == 0) {
            mlvl_anchors_ = all_anchors;
        } else {
            mlvl_anchors_ = torch::cat({mlvl_anchors_, all_anchors}, 0);
        }
    }
}

void SingleStage::get_featuremaps_sizes(){
    feature_maps_.clear();
    feature_maps_.resize(anchor_strides_.size());
    for(int i =0 ; i < int(anchor_strides_.size()); i++) {
        feature_maps_[i].push_back(ceil(float(net_height_) / anchor_strides_[i]));
        feature_maps_[i].push_back(ceil(float(net_width_) / anchor_strides_[i]));
    }
}

void SingleStage::get_maximum_scores(torch::Tensor& output, int digit_nums) {
    assert(anchor_generators_.size() == feature_maps_.size());
    assert(anchor_generators_.size() > 0);
    int start = 0;
    int end = 0;
    torch::Tensor max_output;
    torch::Tensor max_anchor;
    for (int k = 0; k < int(anchor_generators_.size()); k++) {
        int anchor_num = anchor_generators_[k].anchor_nums_;
        end = start + anchor_num;
        torch::Tensor anchors_layer = mlvl_anchors_.slice(0, start, end);
        torch::Tensor output_layer =output.slice(0, start, end);
        torch::Tensor score_layer = output_layer.slice(1, 4, digit_nums);
        if (nms_pre_ > 0 && anchor_num > nms_pre_){
            if (use_sigmoid_ == 0){
                score_layer = score_layer.slice(1, 1, digit_nums);
            }
            std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(score_layer, 1);
            torch::Tensor max_scores= std::get<0>(max_classes);
            std::tuple<torch::Tensor, torch::Tensor> topk = torch::topk(max_scores, nms_pre_, 0);
            torch::Tensor topk_inds = std::get<1>(topk);
            torch::Tensor max_output_layer = output_layer.index_select(0, topk_inds);
            torch::Tensor max_anchors_layer = anchors_layer.index_select(0, topk_inds);
            if (k == 0) {
                max_output = max_output_layer;
                max_anchor = max_anchors_layer;
            } else {
                max_output = torch::cat({max_output, max_output_layer}, 0);
                max_anchor = torch::cat({max_anchor, max_anchors_layer}, 0);
            }
        } else {
            if (k == 0) {
                max_output = output_layer;
                max_anchor = anchors_layer;
            } else {
                max_output = torch::cat({max_output, output_layer}, 0);
                max_anchor = torch::cat({max_anchor, anchors_layer}, 0);
            }
        }
        start = end;
    }
    assert(max_anchor.sizes()[0] == max_output.sizes()[0]);
    output = max_output;
    mlvl_anchors_ = max_anchor;
}


void SingleStage::Detect_Single(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) {
    torch::Tensor tensor_image;
    transform(image, tensor_image, transform_params_, net_width_, net_height_, device_);

    get_featuremaps_sizes();

    get_anchor_boxes();

    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor output = module_->forward({tensor_image}).toTensor().squeeze(0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "forward taken : " << duration.count() << " ms" << std::endl;


    torch::Tensor loc = output.slice(1, 0, 4);
    int digit_nums = output.size(1);
    torch::Tensor mlvl_cores = output.slice(1, 4, digit_nums);
    if (use_sigmoid_ == 0) {
        mlvl_cores = mlvl_cores.softmax(-1);
    } else {
        mlvl_cores.sigmoid_();
    }

    if (nms_pre_ >0 ) {
        get_maximum_scores(output, digit_nums);
        std::cout << output.sizes() << " " << mlvl_anchors_.sizes() << std::endl;
        std::cout << output.slice(0, 0 ,100)  << std::endl;
        std::cout << mlvl_anchors_.slice(0, 0 ,100) << std::endl;
    }

    torch::Tensor mlvl_bboxes = delta2bbox(mlvl_anchors_, loc, {net_width_, net_height_}, target_means_, target_stds_);
    torch::Tensor nms_results = multiclass_nms(mlvl_bboxes, mlvl_cores, scores_thresh_, nms_thresh_, max_num_);

    cv::Size2f scale = cv::Size2f(image.size().width / float(net_width_), image.size().height / float(net_height_));
    bbox2result(nms_results, conf_thresh_, scale, detected_boxes);
}


