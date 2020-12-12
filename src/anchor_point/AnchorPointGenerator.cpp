//
// Created by dl on 2020/4/12.
//

#include "AnchorPointGenerator.hpp"

AnchorPointGenerator::AnchorPointGenerator(int base_size, const std::vector<float>& scales, const std::vector<float>& ratios,
                                 bool scale_major, const std::vector<float>& ctr) {
    base_size_ = base_size;
    scales_ = torch::tensor(scales);
    ratios_ = torch::tensor(ratios);
    scale_major_ = scale_major;
    ctr_  = ctr;
    anchor_nums_ = 0;

    gen_base_anchors();

}

AnchorPointGenerator::AnchorPointGenerator() {}

AnchorPointGenerator::~AnchorPointGenerator() {

}

void AnchorPointGenerator::gen_base_anchors(){
    int w = base_size_;
    int h = base_size_;
    float x_ctr;
    float y_ctr;
    if (ctr_.empty()) {
        x_ctr = 0 * w;
        y_ctr = 0 * h;
    } else {
        x_ctr = ctr_[0];
        y_ctr = ctr_[1];
    }
    torch::Tensor h_ratios = torch::sqrt(ratios_);
    torch::Tensor w_ratios = 1 / h_ratios;

    torch::Tensor ws;
    torch::Tensor hs;

    if (scale_major_) {
        ws = (w * torch::unsqueeze(w_ratios, 1) * torch::unsqueeze(scales_, 0)).view(-1);
        hs = (h * torch::unsqueeze(h_ratios, 1) * torch::unsqueeze(scales_, 0)).view(-1);
    } else {
        ws = (w * torch::unsqueeze(scales_, 1) * torch::unsqueeze(w_ratios, 0)).view(-1);
        hs = (h * torch::unsqueeze(scales_, 1) * torch::unsqueeze(h_ratios, 0)).view(-1);
    }

    base_anchors_ = torch::stack({x_ctr - 0.5 * ws, y_ctr - 0.5 * hs, x_ctr + 0.5 * ws, y_ctr + 0.5 * hs}, 1);
}

torch::Tensor AnchorPointGenerator::grid_anchors(int stride, torch::DeviceType device) {
    base_anchors_ = base_anchors_.to(device);
    int feat_h = feature_maps_sizes_[0];
    int feat_w = feature_maps_sizes_[1];

    torch::Tensor shift_x = torch::arange(0, feat_w, device = device) * stride;
    torch::Tensor shift_y = torch::arange(0, feat_h, device = device) * stride;

    // shift_y is the rows, shiftx is the col
    std::vector<torch::Tensor> args = torch::meshgrid({shift_y, shift_x});
    /********
     * args[0]  0 0 0
     *          1 1 1
     *          2 2 2
     * args[1]  0 1 2
     *          0 1 2
     *          0 1 2
     */
    // args[0]是y的坐标变换，args[1]是x的坐标变换
     torch::Tensor cy = args[0].contiguous().view({-1});
     torch::Tensor cx = args[1].contiguous().view({-1});
     torch::Tensor shifts = torch::stack({cx, cy, cx, cy}, 1);
     torch::Tensor all_anchors = (base_anchors_.unsqueeze(0) + shifts.unsqueeze(1)).view({-1 ,4});

     anchor_nums_ = all_anchors.sizes()[0];
     return all_anchors;
}

torch::Tensor AnchorPointGenerator::grid_points(int feat_h, int feat_w, int stride, torch::DeviceType device) {
    torch::Tensor shift_x = torch::arange(0, feat_w, device = device) * stride;
    torch::Tensor shift_y = torch::arange(0, feat_h, device = device) * stride;
    std::vector<torch::Tensor> args = torch::meshgrid({shift_y, shift_x});

    torch::Tensor cy = args[0].contiguous().view({-1});
    torch::Tensor cx = args[1].contiguous().view({-1});

    torch::Tensor shifts = torch::stack({cx, cy}, 1) + stride / 2;
//    std::cout << shifts[50] << std::endl;
    return shifts;
}
