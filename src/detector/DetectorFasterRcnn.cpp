#include "DetectorFasterRcnn.hpp"

DetectorFasterRcnn::DetectorFasterRcnn()
{

}

DetectorFasterRcnn::~DetectorFasterRcnn(){

}

void DetectorFasterRcnn::LoadParams(const Params& params, torch::DeviceType* device_type){
    LoadCommonParams(params, device_type);
    single_roi_extractor_.init_params(roi_extractor_params_);
    get_anchor_generators({}, rpn_head_params_.anchor_scales_, rpn_head_params_.anchor_ratios_);
}


void DetectorFasterRcnn::LoadTracedModule() {
    LoadCommonTracedModule();
    if (with_shared_) {
        shared_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(shared_module_path_));
    }
    bbox_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(bbox_module_path_));
}


void DetectorFasterRcnn::split_rpn_fpn(const c10::IValue& output) {
    rpn_cls_score_ = output.toTuple()->elements()[0].toTensor();
    rpn_bbox_pred_ = output.toTuple()->elements()[1].toTensor();
    torch::Tensor fpn_data = output.toTuple()->elements()[2].toTensor();

    int start = 0;
    int end = 0;
    int fpn_channels = fpn_params_.out_channels_;
    for(int k = 0; k < int(anchor_generators_.size()); k++) {
      int feature_height = anchor_generators_[k].feature_maps_sizes_[0];
      int feature_width = anchor_generators_[k].feature_maps_sizes_[1];
      start = end;
      int fpn_num_layer = feature_height * feature_width * fpn_channels;
      end = end + fpn_num_layer;
      torch::Tensor fpn_data_layer = fpn_data.slice(0, start, end).view({1, fpn_channels, feature_height, feature_width});
      fpn_datas_.push_back(fpn_data_layer);
    }
}

void DetectorFasterRcnn::get_bboxes(const c10::IValue& output_data,
                                    torch::Tensor& proposals_bboxes,
                                    torch::Tensor&  proposals_scores) {

    split_rpn_fpn(output_data);

    assert(anchor_generators_.size() > 0);

    if (use_sigmoid_ == 1) {
        rpn_cls_score_.sigmoid_();
    }

    torch::Tensor bbox_pred, anchors, ids;
    int start = 0;
    int end = 0;
    for (int k = 0; k < int(strides_.size()); k++) {
        int anchor_num = anchor_generators_[k].anchor_nums_;
        end = start + anchor_num;
        torch::Tensor anchors_layer = mlvl_anchors_.slice(0, start, end);

        torch::Tensor score_layer = rpn_cls_score_.slice(0, start, end);
        torch::Tensor bbox_pred_layer = rpn_bbox_pred_.slice(0, start, end);

        if (nms_pre_ > 0 && anchor_num > nms_pre_){

            std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(score_layer, 1);
            torch::Tensor max_scores= std::get<0>(max_classes);
            std::tuple<torch::Tensor, torch::Tensor> topk = torch::topk(max_scores, nms_pre_, 0);
            torch::Tensor topk_inds = std::get<1>(topk);

            anchors_layer = anchors_layer.index_select(0, topk_inds);
            bbox_pred_layer = bbox_pred_layer.index_select(0, topk_inds);
            score_layer = score_layer.index_select(0, topk_inds);
        }

        torch::Tensor ids_layer = (torch::ones({score_layer.size(0)}) * k).to(bbox_pred_layer);
        if( k == 0) {
            bbox_pred = bbox_pred_layer;
            proposals_scores = score_layer;
            anchors = anchors_layer;
            ids = ids_layer;

        } else {
            bbox_pred = torch::cat({bbox_pred, bbox_pred_layer}, 0);
            proposals_scores = torch::cat({proposals_scores, score_layer}, 0);
            anchors = torch::cat({anchors, anchors_layer}, 0);
            ids = torch::cat({ids, ids_layer}, 0);
        }
        start = end;
    }

    proposals_bboxes = delta2bbox(anchors, bbox_pred, transform_params_.img_shape_, anchor_head_params_.target_means_, anchor_head_params_.target_stds_);

    proposals_bboxes = proposals_bboxes.reshape({-1, 4});
    proposals_scores = proposals_scores.reshape({-1});
    ids = ids.reshape({-1});
    torch::Tensor keep;
    batched_nms(proposals_bboxes, proposals_scores, ids, keep, nms_thresh_, 0);

    int num = std::min(rpn_head_params_.nms_post_, int(keep.size(0)));
    keep = keep.slice(0, 0, num);

    proposals_bboxes = proposals_bboxes.index_select(0, keep);
    proposals_scores = proposals_scores.index_select(0, keep);
    assert(proposals_bboxes.sizes()[0] == proposals_scores.sizes()[0]);

}

void DetectorFasterRcnn::second_stage( torch::Tensor& proposals,
                                       torch::Tensor& bboxes,
                                       torch::Tensor& scores) {

    torch::Tensor rois = bbox2roi(proposals);

    torch::Tensor roi_feats = single_roi_extractor_.bbox_roi_extractor(fpn_datas_, rois);

    auto bbox_head_output = bbox_module_->forward({roi_feats}).toTuple();
    torch::Tensor cls_score = bbox_head_output->elements()[0].toTensor();
    torch::Tensor bbox_pred = bbox_head_output->elements()[1].toTensor();

    scores = torch::softmax(cls_score, 1);

    bboxes = delta2bbox(rois.slice(1, 1, 5), bbox_pred, transform_params_.img_shape_,
                        roi_extractor_params_.target_means_,
                        roi_extractor_params_.target_stds_);
}

void DetectorFasterRcnn::Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) {

    DetectTwoStage(image, detected_boxes);

}
