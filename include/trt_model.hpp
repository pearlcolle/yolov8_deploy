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
        image_info img = {224,224,3};  //已经预处理好的大小
        int  ws_size  = WORKSPACESIZE;
    };



    //创建一个模板
    template<class T>
    void destroy_ptr(T*ptr)
    {
        if(ptr)
        {
            std::string ptr_type = typeid(T).name();
            LOGD("destroy %s",ptr_type.c_str());
            ptr->destroy();
        };
    }




    class Model{



        public:
        Model(std::string onnx_path,logger::Level level,Params params);//构造函数接受onnx路径，打印的水平，以及基本参数来进行初始化
        virtual ~Model() =default;

        //我们总共需要这几部 加载图片，判断model状态（有engine直接用，没engine就是需要build），推理
        void load_image(std::string image_path);
        void init_model(); 
        void inference(); //描述整体架构
        std::string getPrec(precision prec);

        public:

        //如果没有现成的engine，我们需要去build

        bool build_engine();

        //到这里engine肯定有了，所以要loadengine
        bool load_engine();

        void save_plan(nvinfer1::IHostMemory &plan);//保存生成的engine

        void print_network(nvinfer1::INetworkDefinition &network, bool optimized);//打印框架

        bool enqueue_bindings();//进行推理

        //对于以上的函数都是不同任务通用的函数从加载图片，生成engine，推理都是公用的

        //一下函数都是一些不通用的


        // setup负责分配host/device的memory, bindings, 以及创建推理所需要的上下文。
        // 由于不同task的input/output的tensor不一样，所以这里的setup需要在子类实现
        virtual void setup(void const* data, std::size_t size)=0;


        virtual void reset_task() = 0;//检测的时候用到

    // 不同的task的前处理/后处理是不一样的，所以具体的实现放在子类
         virtual bool preprocess_cpu()      = 0;
         virtual bool preprocess_gpu()      = 0;
         virtual bool postprocess_cpu()     = 0;
         virtual bool postprocess_gpu()     = 0;

         //实现成员函数
         public:

         //四条路径

        std::string m_imagePath;
        std::string m_outputPath;
        std::string m_onnxPath;
        std::string m_enginePath;

        cv::Mat m_inputImage;
        Params m_params;

        int    m_workspaceSize;

        //这里我们把显存放入类中，避免重复申请，
        float* m_bindings[2];//捆绑input和output
        float* m_inputMemory[2];
        float* m_outputMemory[2];

        nvinfer1::Dims m_inputDims; //输入维度
        nvinfer1::Dims m_outputDims; //输出维度
        cudaStream_t   m_stream;  //流，确保预处理（Preprocess）、推理（Enqueue）、后处理（Postprocess）在同一流里执行


       //把推理的这些用到的东西放到类中，就可以重复使用了
        std::shared_ptr<logger::Logger>               m_logger;  
        std::shared_ptr<timer::Timer>                 m_timer;
        std::shared_ptr<nvinfer1::IRuntime>           m_runtime;
        std::shared_ptr<nvinfer1::ICudaEngine>        m_engine;
        std::shared_ptr<nvinfer1::IExecutionContext>  m_context;
        std::shared_ptr<nvinfer1::INetworkDefinition> m_network;

    };

}  // namespace model


#endif //__TRT_MODEL_HPP__