#ifndef DETECTORFASTERRCNN_HPP
#define DETECTORFASTERRCNN_HPP

#include "DetectorImpl.hpp"
#include "DetectorCommon.hpp"
#include "SingleRoIExtractor.hpp"

class DetectorFasterRcnn: public DetectorCommon , public DetectorImpl
{
public:
    DetectorFasterRcnn();

    virtual ~DetectorFasterRcnn();

    void LoadParams(const Params& params, torch::DeviceType* device_type) override ;
    void LoadTracedModule() override;
    void Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) override ;

private:
    void second_stage(torch::Tensor& proposals,
                      torch::Tensor& bboxes,
                      torch::Tensor& scores);

    void get_bboxes(const c10::IValue& output_data,
                    torch::Tensor& proposals_bboxes,
                    torch::Tensor& proposals_scores);

    void split_rpn_fpn(const c10::IValue& output_data);



private:
    SingleRoIExtractor single_roi_extractor_;

    torch::Tensor rpn_cls_score_;
    torch::Tensor rpn_bbox_pred_;
    std::vector<torch::Tensor> fpn_datas_;
};

#endif // DETECTORFASTERRCNN_HPP
