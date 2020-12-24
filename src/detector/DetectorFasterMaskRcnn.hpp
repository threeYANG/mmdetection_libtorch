#ifndef DETECTORFASTERMASKRCNN_HPP
#define DETECTORFASTERMASKRCNN_HPP

#include "DetectorImpl.hpp"
#include "DetectorCommon.hpp"
#include "SingleRoIExtractor.hpp"

class DetectorFasterMaskRcnn: public DetectorCommon , public DetectorImpl
{
public:
    DetectorFasterMaskRcnn();

    virtual ~DetectorFasterMaskRcnn();

    void LoadParams(const Params& params, torch::DeviceType* device_type) override ;
    void LoadTracedModule() override;
    void Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) override ;

private:
    void second_stage(const torch::Tensor& proposals,
                      torch::Tensor& bbox_results,
                      torch::Tensor& segm_results);

    void get_bboxes(const c10::IValue& output_data,
                    torch::Tensor& proposals_bboxes,
                    torch::Tensor& proposals_scores);

    void get_segm_masks(const torch::Tensor& bbox_result,
                       torch::Tensor& segm_result);

    void get_mask_pred(const torch::Tensor& bbox_,
                       torch::Tensor& mask_preds);

    void do_paste_mask(const torch::Tensor& mask_pred,
                       const torch::Tensor& bboxes,
                       torch::Tensor& img_masks,
                       int img_h, int img_w);


    void split_rpn_fpn(const c10::IValue& output_data);



private:
    ROIHeadParams roi_head_params_;
    SingleRoIExtractor bbox_roi_;
    SingleRoIExtractor mask_roi_;

    torch::Tensor rpn_cls_score_;
    torch::Tensor rpn_bbox_pred_;
    std::vector<torch::Tensor> fpn_datas_;

    std::unique_ptr<torch::jit::script::Module> bbox_module_;
    std::unique_ptr<torch::jit::script::Module> mask_module_;
};

#endif // DETECTORFASTERMASKRCNN_HPP
