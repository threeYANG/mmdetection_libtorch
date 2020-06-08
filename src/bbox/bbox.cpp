//
// Created by dl on 2020/4/15.
//

#include "bbox.hpp"
#include "nms_kernel.h"
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")


at::Tensor nms(const at::Tensor& dets, const float threshold) {
  CHECK_CUDA(dets);
  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  return nms_cuda(dets, threshold);
}

torch::Tensor delta2bbox(const torch::Tensor& rois, const torch::Tensor& deltas,
                         const std::vector<int>& max_shape,
                         const std::vector<float>& means, const std::vector<float>& stds,
                         float wh_ratio_clip) {

    assert (rois.sizes() == deltas.sizes());
    assert (rois.device() == deltas.device());

    torch::Tensor Tmeans = torch::tensor(means).to(deltas.device()).type_as(deltas);
    Tmeans = Tmeans.repeat({1, deltas.sizes()[1] / 4});
    torch::Tensor Tstds = torch::tensor(stds).to(deltas.device()).type_as(deltas);
    Tstds = Tstds.repeat({1, deltas.sizes()[1] / 4});

    double max_ratio = abs(log(wh_ratio_clip));
    torch::Tensor denorm_deltas = deltas * Tstds + Tmeans;
    torch::Tensor dx = denorm_deltas.select(1, 0);
    torch::Tensor dy = denorm_deltas.select(1, 1);
    torch::Tensor dw = denorm_deltas.select(1, 2).clamp(-max_ratio, max_ratio);
    torch::Tensor dh = denorm_deltas.select(1, 3).clamp(-max_ratio, max_ratio);

    torch::Tensor px = ((rois.select(1, 0) + rois.select(1, 2)) * 0.5).expand_as(dx);
    torch::Tensor py = ((rois.select(1, 1) + rois.select(1, 3)) * 0.5).expand_as(dy);
    torch::Tensor pw = ((rois.select(1, 2) - rois.select(1, 0)) + 1.0).expand_as(dw);
    torch::Tensor ph = ((rois.select(1, 3) - rois.select(1, 1)) + 1.0).expand_as(dh);

    torch::Tensor gw = pw * dw.exp();
    torch::Tensor gh = ph * dh.exp();

    torch::Tensor gx = px + 1 * pw * dx;
    torch::Tensor gy = py + 1 * ph * dy;

    torch::Tensor x1 = (gx - gw * 0.5 + 0.5).clamp(0.0, max_shape[0] - 1);
    torch::Tensor y1 = (gy - gh * 0.5 + 0.5).clamp(0.0, max_shape[1] - 1);
    torch::Tensor x2 = (gx + gw * 0.5 - 0.5).clamp(0.0, max_shape[0] - 1);
    torch::Tensor y2 = (gy + gh * 0.5 - 0.5).clamp(0.0, max_shape[1] - 1);

    return torch::stack({x1, y1, x2, y2}, 1);
}

torch::Tensor multiclass_nms(const torch::Tensor& multi_bboxes,
                             const torch::Tensor& multi_scores,
                             float score_thr, float iou_thr,
                             int max_num) {
    bool first = true;
    torch::Tensor bboxes;
    torch::Tensor labels;
    int num_classes = multi_scores.sizes()[1];
    for(int i = 1; i < num_classes; i++) {
        torch::Tensor cls_inds= multi_scores.select(1, i) > score_thr;
        if (torch::any(cls_inds).item().toBool() == 0) {
            continue;
        }
        cls_inds = torch::nonzero(cls_inds).squeeze();
        torch::Tensor _bboxes = multi_bboxes.index_select(0, cls_inds);
        torch::Tensor _scores = multi_scores.index_select(0, cls_inds).select(1, i).unsqueeze(1);
        torch::Tensor cls_dets = torch::cat({_bboxes, _scores}, 1);
        torch::Tensor inds = nms(cls_dets, iou_thr);
        cls_dets = cls_dets.index_select(0, inds);
        torch::Tensor cls_labels = multi_bboxes.new_full({cls_dets.sizes()[0], 1}, i-1);

        if (first) {
            bboxes = cls_dets;
            labels = cls_labels;
            first = false;
        } else {
            bboxes = torch::cat({bboxes, cls_dets}, 0);
            labels = torch::cat({labels, cls_labels}, 0);
        }
    }
    if (first) {
        bboxes = torch::zeros({0, 5});
        labels = torch::zeros({0, 1});
    } else {
        if (bboxes.sizes()[0] > max_num) {
            bboxes = bboxes.slice(0, 0 , max_num);
            labels = labels.slice(0, 0, max_num);
        }
    }
    return torch::cat({bboxes, labels}, 1);
}

torch::Tensor singleclass_nms(const torch::Tensor& proposals,float iou_thr) {

    torch::Tensor inds = nms(proposals, iou_thr);
    return inds;
}

void bbox2result(torch::Tensor& result, float thresh, cv::Size2f scale,
                 std::vector<DetectedBox>& detected_boxes) {

    if (result.sizes()[0] == 0) {
        return;
    }
    result = result.cpu();
    // Return a `TensorAccessor` for CPU `Tensor`s. You have to specify scalar type and
    auto result_data = result.accessor<float, 2>();
    for (size_t i = 0; i < result.size(0) ; i++)
    {
        float score = result_data[i][4];
        if (score > thresh) {
            DetectedBox detected_box;
            detected_box.box.x = result_data[i][0] * scale.width ;
            detected_box.box.y = result_data[i][1] * scale.height  ;
            detected_box.box.width = (result_data[i][2] - result_data[i][0]) * scale.width;
            detected_box.box.height = (result_data[i][3] - result_data[i][1]) * scale.height;
            detected_box.label = result_data[i][5];
            detected_box.score = score;
            detected_boxes.emplace_back(detected_box);
        }
    }
}


//only one image
torch::Tensor bbox2roi(const torch::Tensor& proposals) {
    torch::Tensor img_inds = proposals.new_full({proposals.sizes()[0],1}, 0);
    torch::Tensor rois = torch::cat({img_inds, proposals.slice(1, 0, 4)}, 1);
    return rois;
}










