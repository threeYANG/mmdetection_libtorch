//
// Created by dl on 2020/4/15.
//

#include "bbox.hpp"


torch::Tensor distance2bbox(const torch::Tensor points, const torch::Tensor distance,
                            const std::vector<int>& max_shape) {
    torch::Tensor x1 = points.select(1, 0) - distance.select(1, 0);
    torch::Tensor y1 = points.select(1, 1) - distance.select(1, 1);
    torch::Tensor x2 = points.select(1, 0) + distance.select(1, 2);
    torch::Tensor y2 = points.select(1, 1) + distance.select(1, 3);
    if (max_shape.size() != 0) {
        x1.clamp_(0.0, max_shape[0] - 1);
        y1.clamp_(0.0, max_shape[1] - 1);
        x2.clamp_(0.0, max_shape[0] - 1);
        y2.clamp_(0.0, max_shape[1] - 1);
    }
    return torch::stack({x1, y1, x2, y2}, 1);

}
torch::Tensor delta2bbox(const torch::Tensor& rois, const torch::Tensor& deltas,
                         const std::vector<int>& max_shape,
                         const std::vector<float>& means, const std::vector<float>& stds,
                         float wh_ratio_clip) {
    torch::Tensor Tmeans = torch::tensor(means).to(deltas.device()).type_as(deltas);
    Tmeans = Tmeans.repeat({1, deltas.sizes()[1] / 4});
    torch::Tensor Tstds = torch::tensor(stds).to(deltas.device()).type_as(deltas);
    Tstds = Tstds.repeat({1, deltas.sizes()[1] / 4});
    double max_ratio = abs(log(wh_ratio_clip));
    torch::Tensor denorm_deltas = deltas * Tstds + Tmeans;


    denorm_deltas = denorm_deltas.view({denorm_deltas.size(0), -1, 4});
    torch::Tensor dx = denorm_deltas.select(2, 0);
    torch::Tensor dy = denorm_deltas.select(2, 1);
    torch::Tensor dw = denorm_deltas.select(2, 2).clamp(-max_ratio, max_ratio);
    torch::Tensor dh = denorm_deltas.select(2, 3).clamp(-max_ratio, max_ratio);

    torch::Tensor px = ((rois.select(1, 0) + rois.select(1, 2)) * 0.5).unsqueeze_(1).expand_as(dx);
    torch::Tensor py = ((rois.select(1, 1) + rois.select(1, 3)) * 0.5).unsqueeze_(1).expand_as(dy);
    torch::Tensor pw = (rois.select(1, 2) - rois.select(1, 0)).unsqueeze_(1).expand_as(dw);
    torch::Tensor ph = (rois.select(1, 3) - rois.select(1, 1)).unsqueeze_(1).expand_as(dh);


    torch::Tensor gw = pw * dw.exp();
    torch::Tensor gh = ph * dh.exp();
    torch::Tensor gx = px + pw * dx;
    torch::Tensor gy = py + ph * dy;

    torch::Tensor x1 = (gx - gw * 0.5).clamp(0.0, max_shape[0] - 1);
    torch::Tensor y1 = (gy - gh * 0.5).clamp(0.0, max_shape[1] - 1);
    torch::Tensor x2 = (gx + gw * 0.5).clamp(0.0, max_shape[0] - 1);
    torch::Tensor y2 = (gy + gh * 0.5).clamp(0.0, max_shape[1] - 1);

    return torch::stack({x1, y1, x2, y2}, -1).view(deltas.sizes());
}


void bbox2result(torch::Tensor& result, float thresh,
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
            detected_box.box.x = result_data[i][0];
            detected_box.box.y = result_data[i][1];
            detected_box.box.width = result_data[i][2] - result_data[i][0];
            detected_box.box.height = result_data[i][3] - result_data[i][1];
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










