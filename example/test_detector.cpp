#include "types.hpp"
#include "Detector.hpp"


int main() {
    /**********
    torch::Tensor a = torch::randint(1, 30, {8});
    std::cout << a << std::endl;
    std::vector<torch::Tensor> index = torch::where( a >= 10);
    std::cout << index[0] <<std::endl;

    torch::Tensor c = torch::randint(35, 50, {8, 3, 4});
    std::cout << c <<std::endl;
    torch::Tensor b = torch::rand({index[0].size(0), 3, 4});
    std::cout << b <<std::endl;

    c.index_copy_(0, index[0], b);
    std::cout << c <<std::endl;

    torch::Tensor mask = a == 3;
    std::cout << mask <<std::endl;
    torch::Tensor index_1 = mask.nonzero();
    std::cout << index_1 <<std::endl;
    ***********/


    //std::string config_file = "/home/dl/Project/mmdetection_libtorch/config/config_ssd.json";
    //std::string config_file = "/home/dl/Project/mmdetection_libtorch/config/config_re.json";
    //std::string config_file = "/home/dl/Project/mmdetection_libtorch/config/config_fcos.json";
    std::string config_file = "/home/dl/Project/mmdetection_libtorch/config/config_faster_rcnn.json";
    Params params;
    long res = params.Read(config_file);
    if (res == -1) {
        return -1;
    }


    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    Detector detector;
    res = detector.Create(params.detector_type_);
    if (res < 0) {
        return -1;
    }
    detector.LoadParams( params, &device_type);
    detector.LoadTracedModule();

    cv::Mat image = cv::imread("/home/dl/Project/mmdetection_libtorch/image/000012.jpg");
    std::vector<DetectedBox> detected_boxes;
    res = detector.Detect(image, detected_boxes);
    std::cout << detected_boxes.size() << std::endl;
    if (res == 0) {
        for(int i = 0; i < detected_boxes.size(); i++) {
            std::cout << detected_boxes[i].box.x <<" ";
            std::cout << detected_boxes[i].box.y << " ";
            std::cout << detected_boxes[i].box.x + detected_boxes[i].box.width << " ";
            std::cout << detected_boxes[i].box.y + detected_boxes[i].box.height <<" ";
            std::cout << detected_boxes[i].score <<std::endl;
            cv::rectangle(image, detected_boxes[i].box, cv::Scalar(0, 0, 255), 1, 1, 0);
        }
        cv::imwrite("/home/dl/Project/mmdetection_libtorch/result.jpg", image);
    }

    return 0;

}

