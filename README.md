# mmdetection_libtorch
&#8195; &#8195;本来打算英文写Readme的，奈何菜鸟一枚，英文表达能力太差，自己写完自己都有点看不懂，呵呵，于是乎，写个中文版的，再写个英文版的，中文版是呼之欲出，英文版是一改再该，半天拿不出手，等改好再发吧。（by the way，好像中文版Readme是很掉档啊，哈哈哈）

&#8195; &#8195;发现mmdetection在工作中，用来尝试各种网络框架还是挺方便的，也便于在细节中动手脚，但是实际部署的时候用的是libtorch，于是乎，找了几个具有代表性的检测框架，将其的libtorch版本给写出来了（其实根本原因是闲的慌，哈哈哈）。

&#8195; &#8195;mmdetection种涉及很多中检测框架，这里进行部署的，只有SSD、Retinanet、FasterRcnn(完成了一半，下面的工作是roi_align，mmdetection使用的是移植到cuda版本的，我一直没有移植成功，就卡到这里了)。后续如果有时候的话，有可能会把FCOS加进去。

## Requirements（这里是我使用的）

 - Python 3.7
 - Libtorch 1.3.1
 - CUDA: 9.0/9.2/10.0/10.1
 -  mmdetection: v1.0
 - mmcv: 0.4.0
 
&#8195; &#8195;这里需要说明的是：目前官方的mmdetection已经出到v2.0版本了，但是我使用的还是v1.0版本，我写这个代码的目的已经达到（编程能力、熟悉检测框架细节），所以官方的v2.0对我来说意义不大，各位客官使用的时候根据个人版本进行修改。
 
## nms_cuda_lib
&#8195; &#8195;mmdetection是python接口，在使用nms的时候，将其移植到cuda上，作为外接库使用，于是乎我将nms的cpp代码拿出来，单独编译为库，链接到我的mmdetection_libtorch工程中进行使用

## roi_align_lib
&#8195; &#8195;和nms同样的使用方式，但是这次我没有编译成功，遇到的问题是，cuda和torch同时重载了==,>,<,等符号，因此编译器不知道使用哪一个，libtorch是从官网上下载的编译好的，知道问题点，但是不知道怎么该，一直没有改成功，导致这个库没有编译成功，faster_rcnn的部署工作就卡在了这里。


## how to trace a trained detector
1、使用mmdetection中的某个网络（ssd、retinanet)等进行训练。（这里需要注意的是，因为trace的时候，无法识别mmdet/opt/conv_module.py中的norm方法，因此这里我注释掉了，但是埋下了一个隐患，用到这个方法的其他网络框架会出现bug）;
2、single_stage.py 和 two_stage.py都加入了forward_trace函数;
3、使用get_trace.py进行trace，将pytorch生成的weights转换为libtorch可以载入的权重格式;
4、mmdetection_libtorch工程中，修改configs中的配置参数。
 
  ## Traced  weights
 [ssd300_voc_traced](https://pan.baidu.com/s/1oE8HuiZv-s7U3SOM-Ede-Q)  4tbe
 这个是我用ssd300, voc数据训练的，可以直接在mmdetection_libtorch工程中使用。
 ## 说明
 1、ssd、retinanet、faster_rcnn三个网络的libtorch的部署中，其数据流和python接口的数据流保持一致，因此c++代码逻辑没有大的问题。

 2、mmdetection_libtorch工程中的各种功能c++代码都是从mmdetection中的python代码中移植过来的，其思路和表达方式都一致。

 3、mmdetection_libtorch工程中的c++代码，由于本人能力的问题，在实际使用的过程中，还可以进行各方面的优化。
