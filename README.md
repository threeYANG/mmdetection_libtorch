# mmdetection_libtorch
&#8195; &#8195;Deploy mmdetection on libtorch，trained on mmdetection and deploy on libtorch，for now only support **single stage(SSD , Retinanet）**，two_stage is in the future。
&#8195; &#8195;As we all know, torch not only has python API, but also C++ API, we use C++ API to help the trained detector using in C++ project。

## Requirements

 - Python 3.5+
 - Libtorch 1.3.1+
 - CUDA: 9.0/9.2/10.0/10.1
 
## nms_cuda_lib
&#8195; &#8195;You need first  use  **nms_kernel.cu 、nms_kernel.h** to  compile a libnms_cuda and add the lib in the mmdetection_libtorch project.


## how to trace a trained detector
 1. add **get_trace.py** in the mmdetection/tools/
 2. modify  *mmdetction/mmdet/models/detectors/single_stage.py*  , add  function **forward_trace(self, img)**
 3. modify *mmdetection/mmdet/models/utils/conv_module.py*, comment the function **def norm(self)**,  because  it is unrecognized when traced
