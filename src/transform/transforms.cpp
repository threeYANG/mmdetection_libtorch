#include "transforms.hpp"


float imrescale(const cv::Mat& image, cv::Mat& image_resize,
              const std::vector<int>& img_scale, int& net_width, int& net_height) {
    float h = image.rows;
    float w = image.cols;

    float max_long_edge = std::max(img_scale[0], img_scale[1]);
    float max_short_edge = std::min(img_scale[0], img_scale[1]);

    float scale_factor = std::min(max_long_edge / std::max(h, w),
                                  max_short_edge / std::min(h, w));

    net_width = w * scale_factor + 0.5;
    net_height = h * scale_factor + 0.5;

    cv::resize(image, image_resize, cv::Size(net_width, net_height));
    return scale_factor;
}


cv::Mat impad(const cv::Mat& image, std::vector<int>& pad_shape, int pad_val) {

    assert(pad_shape[0] >= image.rows);
    assert(pad_shape[1] >= image.cols);
    cv::Mat pad_image = cv::Mat::ones(pad_shape[0], pad_shape[1], image.type()) * pad_val;

    image.copyTo(pad_image(cv::Rect(0, 0, image.cols, image.rows)));

    return pad_image;

}

cv::Mat impad_to_multiple(const cv::Mat& image, int divisor, int pad_val = 0)
{
   int pad_h = int(ceil(float(image.rows) / divisor)) * divisor;
   int pad_w = int(ceil(float(image.cols) / divisor)) * divisor;
   std::vector<int> pad_shape = {pad_h, pad_w};
   return impad(image, pad_shape, pad_val);
}



void normalize(torch::Tensor& tensor_image,
               const std::vector<float>& mean,
               const std::vector<float>& std) {
    tensor_image[0][0].sub_(mean[0]).div_(std[0]);
    tensor_image[0][1].sub_(mean[1]).div_(std[1]);
    tensor_image[0][2].sub_(mean[2]).div_(std[2]);

}

void normalize(cv::Mat& resize_image,
               const std::vector<float>& mean,
               const std::vector<float>& std) {

    std::cout <<"a3: " <<resize_image.rows <<" " <<resize_image.cols << std::endl;
    cv::Vec3b bgr = resize_image.at<cv::Vec3b>(100, 100);
    std::cout << int(bgr[0])<<" "<< int(bgr[1])<<" "<< int(bgr[2])<<std::endl;

    resize_image.convertTo(resize_image, CV_32FC3, 1.0, 0);
    std::cout <<"a4: " <<resize_image.rows <<" " <<resize_image.cols << std::endl;
    cv::Vec3f bgr_f = resize_image.at<cv::Vec3f>(100, 100);
    std::cout << bgr_f[0]<<" "<< bgr_f[1]<<" "<< bgr_f[2]<<std::endl;


    cv::cvtColor(resize_image, resize_image, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> input_channels;
    cv::split(resize_image, input_channels);
    for(int i = 0; i < 3; i++) {
         input_channels[i]= (input_channels[i]-mean[i]) / std[i];
    }
     cv::merge(input_channels, resize_image);
}
void transform(const cv::Mat& image, torch::Tensor& tensor_image,
               TransformParams& transform_params,
               int& net_width, int& net_height,
               torch::DeviceType* device)
{
    std::cout << "a1: " <<image.rows<<" "<<image.cols<< std::endl;
    cv::Vec3b bgr = image.at<cv::Vec3b>(100, 100);
    std::cout << int(bgr.val[0])<<" "<< int(bgr.val[1])<<" "<< int(bgr.val[2])<<std::endl;
    cv::Mat image_resize;
    if (transform_params.keep_ratio_ == 0) {
        net_width = transform_params.img_scale_[0];
        net_height = transform_params.img_scale_[1];
        cv::resize(image, image_resize, cv::Size(net_width, net_height));
    }
    else if (transform_params.keep_ratio_ == 1) {
        transform_params.scale_factor_ = imrescale(image, image_resize, transform_params.img_scale_, net_width, net_height);
    }
    std::cout << "a2: " <<image_resize.rows<<" "<<image_resize.cols<< std::endl;
    bgr = image_resize.at<cv::Vec3b>(100, 100);
    std::cout << int(bgr[0])<<" "<< int(bgr[1])<<" "<< int(bgr[2])<<std::endl;
    transform_params.img_shape_.push_back(image_resize.rows);
    transform_params.img_shape_.push_back(image_resize.cols);
    transform_params.img_shape_.push_back(image_resize.channels());
    assert (image_resize.empty() != 1);

    normalize(image_resize, transform_params.mean_, transform_params.std_);

    std::cout <<"a5: " <<image_resize.rows <<" " << image_resize.cols << std::endl;
    cv::Vec3f bgr_f = image_resize.at<cv::Vec3f>(100, 100);
    std::cout << bgr_f[0]<<" "<< bgr_f[1]<<" "<< bgr_f[2]<<std::endl;
    if (transform_params.pad_ > 0) {
       image_resize = impad_to_multiple(image_resize, transform_params.pad_);
       net_width = image_resize.cols;
       net_height = image_resize.rows;
       std::cout <<"a6: " <<image_resize.rows <<" " << image_resize.cols << std::endl;
       bgr_f = image_resize.at<cv::Vec3f>(100, 100);
       std::cout << bgr_f[0]<<" "<< bgr_f[1]<<" "<< bgr_f[2]<<std::endl;
    }

// 下方的代码即将图像转化为Tensor，随后导入模型进行预
    tensor_image = torch::from_blob(image_resize.data, {1, image_resize.rows, image_resize.cols,3});
    tensor_image = tensor_image.to(*device);
    tensor_image = tensor_image.permute({0,3,1,2});
    tensor_image = tensor_image.toType(torch::kFloat);
}
