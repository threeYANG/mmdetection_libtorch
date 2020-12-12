#include "DetectorFCOS.hpp"

DetectorFCOS::DetectorFCOS() {
    detector_type_ = DetetorType::FCOS;
}

DetectorFCOS::~DetectorFCOS() {

}

void DetectorFCOS::LoadParams(const Params& params, torch::DeviceType* device_type) {

    LoadCommonParams(params, device_type);

    point_generator_ = AnchorPointGenerator();
}

void DetectorFCOS::LoadTracedModule(){

    LoadCommonTracedModule();
}



void DetectorFCOS::get_anchors() {

    for (int i = 0; i < strides_.size(); i++) {
        int stride = strides_[i];

        int feat_h = ceil(float(net_height_) / stride);
        int feat_w = ceil(float(net_width_) / stride);
        mlvl_points_.push_back(point_generator_.grid_points(feat_h, feat_w, stride, *device_));
    }
}


void DetectorFCOS::get_bboxes(const c10::IValue& output_data,
                             torch::Tensor& bboxes,
                             torch::Tensor& scores) {
    torch::Tensor cls_score = output_data.toTuple()->elements()[0].toTensor();
    torch::Tensor bbox_pred = output_data.toTuple()->elements()[1].toTensor();
    torch::Tensor centerness_pred = output_data.toTuple()->elements()[2].toTensor();

    torch::Tensor centerness;
    int start = 0;
    int end = 0;
    for (int k = 0; k < int(strides_.size()); k++) {

        int stride = strides_[k];

        int feat_h = ceil(float(net_height_) / stride);
        int feat_w = ceil(float(net_width_) / stride);
        int point_num = feat_h * feat_w;
        end = start + point_num;

        torch::Tensor score_layer =cls_score.slice(0, start, end).sigmoid_();
        torch::Tensor bbox_pred_layer = bbox_pred.slice(0, start, end);
        torch::Tensor centerness_layer = centerness_pred.slice(0, start, end).sigmoid_();
        torch::Tensor points = mlvl_points_[k];

        if (nms_pre_ > 0 && score_layer.size(0) > nms_pre_) {
            std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(score_layer * centerness_layer, 1);
            torch::Tensor max_scores= std::get<0>(max_classes);
            std::tuple<torch::Tensor, torch::Tensor> topk = torch::topk(max_scores, nms_pre_, 0);
            torch::Tensor topk_inds = std::get<1>(topk);

            points = points.index_select(0, topk_inds);
            bbox_pred_layer = bbox_pred_layer.index_select(0, topk_inds);
            score_layer = score_layer.index_select(0, topk_inds);
            centerness_layer = centerness_layer.index_select(0, topk_inds);
        }

        start = end;

        torch::Tensor bboxes_layer = distance2bbox(points, bbox_pred_layer, transform_params_.img_shape_);

        if( k == 0) {
            bboxes = bboxes_layer;
            scores = score_layer;
            centerness = centerness_layer;

        } else {
            bboxes = torch::cat({bboxes, bboxes_layer}, 0);
            scores = torch::cat({scores, score_layer}, 0);
            centerness = torch::cat({centerness, centerness_layer}, 0);
        }
    }
    scores = scores * centerness;
}


void DetectorFCOS::Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) {

    DetectOneStage(image, detected_boxes);

}







