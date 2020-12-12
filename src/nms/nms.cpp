#include "pytorch_cpp_helper.hpp"

at::Tensor NMSCUDAKernelLauncher(at::Tensor boxes, at::Tensor scores, float iou_threshold,
                             int offset);

at::Tensor nms_cuda(at::Tensor boxes, at::Tensor scores, float iou_threshold, int offset) {
  return NMSCUDAKernelLauncher(boxes, scores, iou_threshold, offset);
}

at::Tensor nms(at::Tensor boxes, at::Tensor scores, float iou_threshold, int offset) {
    CHECK_CUDA_INPUT(boxes);
    CHECK_CUDA_INPUT(scores);
    return nms_cuda(boxes, scores, iou_threshold, offset);

}

void batched_nms(const torch::Tensor& bboxes, const torch::Tensor& scores,
                 const torch::Tensor& idxs, torch::Tensor& keep,
                 float iou_thr, bool class_agnostic) {
    torch::Tensor bboxes_for_nms;
    if (class_agnostic) {
        bboxes_for_nms = bboxes;
    } else {
        torch::Tensor max_coordinate = bboxes.max();
        torch::Tensor offsets = idxs * (max_coordinate + torch::ones({1}).to(bboxes));

        bboxes_for_nms = bboxes + offsets.unsqueeze_(1).expand({bboxes.size(0), bboxes.size(1)});
    }
    keep = nms(bboxes_for_nms, scores, iou_thr, 0);
}


torch::Tensor multiclass_nms(const torch::Tensor& multi_bboxes,
                             const torch::Tensor& multi_scores,
                             float score_thr, float iou_thr,
                             int max_num) {

   int num_classes = multi_scores.size(1) - 1;
   int num_boxes = multi_scores.size(0);
   torch::Tensor bboxes;
   if (multi_bboxes.sizes()[1] > 4){
       bboxes = multi_bboxes.view({num_boxes, -1, 4});
   } else {
       bboxes = multi_bboxes.unsqueeze_(1).expand({num_boxes, num_classes, 4});
   }

   torch::Tensor scores = multi_scores.slice(1, 0, num_classes);

   torch::Tensor labels = torch::arange(0, num_classes).toType(torch::kLong).cuda();
   labels = labels.view({1, -1}).expand_as(scores);

   bboxes = bboxes.reshape({-1, 4});
   scores = scores.reshape({-1});
   labels = labels.reshape({-1});
   //remove low scoring boxes
   torch::Tensor valid_mask = scores > score_thr;
   torch::Tensor inds = valid_mask.nonzero().squeeze(1);

   torch::Tensor s_bboxes = bboxes.index_select(0,inds);
   torch::Tensor s_scores = scores.index_select(0,inds);
   torch::Tensor s_labels = labels.index_select(0,inds);

   torch::Tensor keep;
   bool class_agnostic = false;
   batched_nms(s_bboxes, s_scores, s_labels, keep, iou_thr, class_agnostic);

   torch::Tensor dets_bboxes = s_bboxes.index_select(0, keep);
   torch::Tensor dets_scores = s_scores.index_select(0, keep).unsqueeze_(1);
   torch::Tensor dets_labels = s_labels.index_select(0, keep).unsqueeze_(1).type_as(dets_bboxes);

   torch::Tensor dets = torch::cat({dets_bboxes, dets_scores, dets_labels}, 1);
   return dets;
}
