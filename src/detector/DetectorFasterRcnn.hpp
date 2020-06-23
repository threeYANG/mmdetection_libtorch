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
    void get_rpn_fpn_data(const torch::Tensor& output, torch::Tensor& rpn_data, std::vector<torch::Tensor>& fpn_datas);


private:
    SingleRoIExtractor single_roi_extractor_;
    std::string bone_model_path_;
    std::string shared_model_path_;
    std::string bbox_model_path_;

    bool with_shared_;
    std::unique_ptr<torch::jit::script::Module> bone_module_;
    std::unique_ptr<torch::jit::script::Module> shared_module_;
    std::unique_ptr<torch::jit::script::Module> bbox_module_;
};

#endif // DETECTORFASTERRCNN_HPP
