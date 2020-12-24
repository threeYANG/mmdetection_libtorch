#include "DetectorFasterMaskRcnn.hpp"

const int BYTES_PER_FLOAT = 4;

const int GPU_MEM_LIMIT = std::pow(1024,3);  // 1 GB memory limit

DetectorFasterMaskRcnn::DetectorFasterMaskRcnn()
{
    bbox_module_ = nullptr;
    mask_module_ = nullptr;

}

DetectorFasterMaskRcnn::~DetectorFasterMaskRcnn(){

}

void DetectorFasterMaskRcnn::LoadParams(const Params& params, torch::DeviceType* device_type){
    LoadCommonParams(params, device_type);
    roi_head_params_ = params.roi_head_params_;
    bbox_roi_.init_params(roi_head_params_.bbox_roi_head_.roi_layer_);
    if (roi_head_params_.with_mask_) {
        mask_roi_.init_params(roi_head_params_.mask_roi_head_.roi_layer_);
    }
    get_anchor_generators({}, rpn_head_params_.anchor_scales_, rpn_head_params_.anchor_ratios_);
}


void DetectorFasterMaskRcnn::LoadTracedModule() {
    LoadCommonTracedModule();
    bbox_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(roi_head_params_.bbox_roi_head_.model_path_));
    if (roi_head_params_.with_mask_) {
        mask_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(roi_head_params_.mask_roi_head_.model_path_));
    }
}


void DetectorFasterMaskRcnn::split_rpn_fpn(const c10::IValue& output) {
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

void DetectorFasterMaskRcnn::get_bboxes(const c10::IValue& output_data,
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

    proposals_bboxes = delta2bbox(anchors, bbox_pred, transform_params_.img_shape_,
                                  anchor_head_params_.target_means_, anchor_head_params_.target_stds_);

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

void DetectorFasterMaskRcnn::get_mask_pred(const torch::Tensor& bbox_,
                                          torch::Tensor& mask_preds) {
    torch::Tensor mask_rois = bbox2roi(bbox_);
    torch::Tensor mask_feats = mask_roi_.roi_extractor(fpn_datas_, mask_rois);
    mask_preds = mask_module_->forward({mask_feats}).toTensor().squeeze(0);
}

/***********
Args:
       masks (Tensor): N, 1, H, W
       boxes (Tensor): N, 4
       img_h (int): Height of the image to be pasted.
       img_w (int): Width of the image to be pasted.
       skip_empty (bool): Only paste masks within the region that
           tightly bound all boxes, and returns the results this region only.
           An important optimization for CPU.

   Returns:
       tuple: (Tensor, tuple). The first item is mask tensor, the second one
           is the slice object.
       If skip_empty == False, the whole image will be pasted. It will
           return a mask of shape (N, img_h, img_w) and an empty tuple.
       If skip_empty == True, only area around the mask will be pasted.
           A mask of shape (N, h', w') and its start and end coordinates
           in the original image will be returned.

   On GPU, paste all masks together (up to chunk size)
   by using the entire image to sample the masks
   Compared to pasting them one by one,
   this has more operations but is faster on COCO-scale dataset.
************/
void  DetectorFasterMaskRcnn::do_paste_mask(const torch::Tensor& mask_pred,
                                            const torch::Tensor& bboxes,
                                            torch::Tensor& img_masks,
                                            int img_h, int img_w) {
    int x0_int = 0;
    int y0_int = 0;
    int x1_int = img_w;
    int y1_int = img_h;

    int N = mask_pred.size(0);

    std::vector<torch::Tensor> split_coordinate = torch::split(bboxes, 1, 1);
    torch::Tensor x0 = split_coordinate[0];
    torch::Tensor y0 = split_coordinate[1];
    torch::Tensor x1 = split_coordinate[2];
    torch::Tensor y1 = split_coordinate[3];

    torch::Tensor img_y = torch::arange(y0_int, y1_int, at::device(*device_).dtype(at::kFloat)) + 0.5;
    torch::Tensor img_x = torch::arange(x0_int, x1_int, at::device(*device_).dtype(at::kFloat)) + 0.5;

    img_y = (img_y - y0) / (y1 - y0) * 2 -1;
    img_x = (img_x - x0) / (x1 - x0) * 2 -1;


    if (torch::isfinite(img_x).any().dim() > 0) {
        std::vector<torch::Tensor> inds = torch::where(torch::isfinite(img_x));
        img_x[inds[0]] = 0;
    }

    if (torch::isfinite(img_x).any().dim() > 0) {
        std::vector<torch::Tensor> inds = torch::where(torch::isfinite(img_y));
        img_y[inds[0]] = 0;
    }


    torch::Tensor gx = img_x.unsqueeze(1).expand({N, img_y.size(1), img_x.size(1)});
    torch::Tensor gy = img_y.unsqueeze(2).expand({N, img_y.size(1), img_x.size(1)});
    torch::Tensor grid = torch::stack({gx, gy}, 3);
    img_masks = torch::grid_sampler(mask_pred.toType(torch::kFloat32), grid, 0, 0, false);
    img_masks.squeeze_();
}


void DetectorFasterMaskRcnn::get_segm_masks(const torch::Tensor& bbox_result,
                                            torch::Tensor& segm_result) {
   torch::Tensor scale_factor = torch::tensor(transform_params_.scale_factor_);
   torch::Tensor bbox_ = bbox_result.slice(1, 0, 4)* scale_factor;
   torch::Tensor det_labels = bbox_result.slice(1, 5, 6);
   torch::Tensor mask_preds;
   get_mask_pred(bbox_, mask_preds);
   mask_preds.sigmoid_();

   int N = int(bbox_result.size(0));
   int img_w = transform_params_.ori_shape_[0];
   int img_h = transform_params_.ori_shape_[1];
   int num_chunks = std::ceil((N * img_w * img_h *BYTES_PER_FLOAT) / float(GPU_MEM_LIMIT));

   std::vector<torch::Tensor> chunks = torch::chunk(torch::arange(0, N).cuda(), num_chunks);

   segm_result = torch::zeros({N, img_h, img_w}).cuda();
   float threshold = roi_head_params_.mask_roi_head_.mask_thr_binary_;
   if (threshold >=0) {
       segm_result = segm_result.toType(torch::kBool);
   } else {
       segm_result = segm_result.toType(torch::kUInt8);
   }
   det_labels = det_labels.toType(torch::kLong).squeeze();
   torch::Tensor mask_pred_select = torch::ones({N,mask_preds.size(2), mask_preds.size(3)}, at::device(*device_));
   for(int i = 0; i < N; i++) {
       mask_pred_select[i] = mask_preds[i][det_labels[i]];
   }
   mask_pred_select.unsqueeze_(1);
   for(int i=0; i < int(chunks.size()); i++) {
       do_paste_mask(mask_pred_select, bbox_result.slice(1, 0, 4),
                     segm_result, img_h, img_w);
       segm_result = segm_result >= threshold;
   }
}


void DetectorFasterMaskRcnn::second_stage( const torch::Tensor& proposals,
                                           torch::Tensor& bbox_results,
                                           torch::Tensor& segm_results) {

    torch::Tensor rois = bbox2roi(proposals);

    torch::Tensor roi_feats = bbox_roi_.roi_extractor(fpn_datas_, rois);

    auto bbox_head_output = bbox_module_->forward({roi_feats}).toTuple();
    torch::Tensor cls_score = bbox_head_output->elements()[0].toTensor();
    torch::Tensor bbox_pred = bbox_head_output->elements()[1].toTensor();

    torch::Tensor scores = torch::softmax(cls_score, 1);

    torch::Tensor bboxes = delta2bbox(rois.slice(1, 1, 5), bbox_pred,
                        transform_params_.img_shape_,
                        roi_head_params_.bbox_roi_head_.target_means_,
                        roi_head_params_.bbox_roi_head_.target_stds_);

    torch::Tensor scale_factor = torch::tensor(transform_params_.scale_factor_);
    bboxes = bboxes / scale_factor;
    bbox_results = multiclass_nms(bboxes, scores, score_thresh_,
                                  nms_thresh_, max_per_img_);

    if (roi_head_params_.with_mask_) {
        get_segm_masks(bbox_results, segm_results);
    }
}

void DetectorFasterMaskRcnn::Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) {

    DetectTwoStage(image, detected_boxes);

}
