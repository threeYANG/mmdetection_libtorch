#include "types.hpp"
#include "Detector.hpp"
#include "cstdlib"

cv::Scalar get_color() {
    srand(int(time(0)));
    cv::Scalar color;
    for(int i = 0; i < 3; i++) {
       color.val[i] = rand() % 255;
    }
    return color;
}
int main() {

    //std::string config_file = "/home/dl/mmdetection_libtorch/config/config_retinanet.json";
    //std::string config_file = "/home/dl/mmdetection_libtorch/config/config_fcos.json";
    //std::string config_file = "/home/dl/mmdetection_libtorch/config/config_fasterrcnn.json";
    std::string config_file = "/home/dl/mmdetection_libtorch/config/config_maskrcnn.json";
    Params params;
    long res = params.Read(config_file);
    if (res == -1) {
        return -1;
    }

    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
    } else {
        std::cout << "not support torch::kCPU" <<std::endl;
        return 0;
     }

    Detector detector;
    res = detector.Create(params.detector_type_);
    if (res < 0) {
        return -1;
    }
    detector.LoadParams(params, &device_type);
    detector.LoadTracedModule();

    cv::Mat image = cv::imread("/home/dl/Project/mmdetection_libtorch/image/000012.jpg");
    std::vector<DetectedBox> detected_boxes;
    res = detector.Detect(image, detected_boxes);


    for(int i = 0; i < detected_boxes.size(); i++)
    {
        std::cout << detected_boxes[i].box.x <<" ";
        std::cout << detected_boxes[i].box.y << " ";
        std::cout << detected_boxes[i].box.x + detected_boxes[i].box.width << " ";
        std::cout << detected_boxes[i].box.y + detected_boxes[i].box.height <<" ";
        std::cout << detected_boxes[i].score <<std::endl;

        if (!detected_boxes[i].seg_mask.empty()) {
            cv::Scalar color = get_color();
            int xmin = detected_boxes[i].box.x;
            int ymin = detected_boxes[i].box.y;
            int xmax = detected_boxes[i].box.x + detected_boxes[i].box.width;
            int ymax = detected_boxes[i].box.y + detected_boxes[i].box.height;
            cv::Mat roi = image(cv::Range(ymin, ymax), cv::Range(xmin, xmax));
            cv::Mat img_temp = roi * 0.5 + color * 0.5;
            img_temp.copyTo(roi, detected_boxes[i].seg_mask);
        }
        cv::rectangle(image, detected_boxes[i].box, cv::Scalar(0, 0, 255), 1, 1, 0);

        cv::imwrite("/home/dl/mmdetection_libtorch/result.jpg", image);
    }
    return 0;

}

