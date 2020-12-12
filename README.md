# mmdetection_libtorch

&#8195; &#8195;mmdetection is very convenient  for us to train all kind of detector, but in the work, after training , we are more willing to use the trained detector in our special project to test results.  So only forward of a detector is developed in libtorch, written in C++ . do not worry, the  c++ code is consistent with python code.










## Requirements
 - Python 3.7
 - Libtorch 1.3.1
 - CUDA: 10.0
 -  mmdetection: v2.7.0 
 - mmcv-full: 01.2.1
 - CMake
 
## Support Module
 1. [x] SSD
 2. [x] FCOS
 3. [x] RetinaNet
 4. [x] Faster R-CNN
 
## How to Train?
reference to   [mmdetection](https://github.com/open-mmlab/mmdetection)
## How to Trace?
 -  the function ***forward_trace_xxx( )*** is  for  data  stream
 - using ***get_trace_one_stage.py***  or ***get_trace_two_stage.py***   to  get traced weights

## How to Test?
 - using your own traced weights  or [download my traced weights](https://pan.baidu.com/s/1g2bQknkPpbmqTbFPXRQTAA)  (9rv5, trained in voc data)
 - modifiy the file  config/xxx.json
 - check that  the config file path  is correct in example/test_detector.cpp
 - just  running the code



