#ifndef __TRT_MODEL_HPP__
#define __TRT_MODEL_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "trt_timer.hpp"
#include "trt_logger.hpp"

#include "trt_preprocess.hpp"


#define WORKSPACESIZE 1<<28



//model是一个大统一，在进行多任务的时候，复用的部分都由model处理，，但是比如前处理，后处理方面等细节需单独处理


//创建一个model的作用域
namespace model{

    //由于model是大一统的类，通用的东西他都需要包含，就比如，什么类型的任务，cpu，gpu
    //精度，图片的大小


    //创建一个枚举，里面去选择你要做的任务类型
    enum task_type{
    CLASSIFICATION,
    DETECTION,
    SEGMENTATION,
    MULTITASK
    };

    enum device{
        GPU,
        CPU
    };

    enum precision{
        FP32,
        FP16,
        INT8
    };

    struct image_info
    {
        int h;
        int w;
        int c;
        image_info(int H,int W,int C):h(H),w(W),c(C)
        {};
    };


    //创建一个结构体储存上面信息，并进行默认
    //上面这些东西无论什么任务都是需要的，所以放在model中再合适不过

    struct Params
    {
        //枚举可以跟类一样，进行初始化，这样在之后我们调用时不在去调用device，而是params.dev
        device dev=GPU;
        precision pre =FP32;
        int num_cls=1000;
        preprocess::tactics tac =preprocess::tactics::GPU_WARP_AFFINE;
        task_type tas = DETECTION;
        image_info img = {224,224,3};

         int      ws_size  = WORKSPACESIZE;
    };
    


    










































































}


#endif //__TRT_MODEL_HPP__