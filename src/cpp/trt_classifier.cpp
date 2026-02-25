#include "opencv2/imgproc.hpp"
#include "trt_model.hpp"
#include "utils.hpp" 
#include "trt_logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "imagenet_labels.hpp"
#include "trt_classifier.hpp"
#include "trt_preprocess.hpp"
#include "utils.hpp"

namespace model{
    namespace classifier{
        void Classifier::setup(void const* data, std::size_t size)
        {

            //进行反序列化准备以及分配内存

            m_runtime = shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_logger),destory_ptr<nvinfer1::IRuntime>);

            m_engine = shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(data,size),destory_ptr<nvinfer1::ICudaEngine>);

            m_context = shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext(),destory_ptr<nvinfer1::IExecutionContext>);

            m_inputDims=m_context->getBindingDimensions(0);
            m_outputDims = m_context->getBindingDimensions(1);


            //创建内存
            m_inputSize = m_params.img.c*m_params.img.w*m_params.img.h*sizeof(float);
            m_outputSize=m_params.num_cls*sizeof(float);
            m_imgArea = m_params.img.w*m_params.img.h;


            //输入在host和dev上的内存
            CUDA_CHECK(cudaMallocHost(&m_inputMemory[0],m_inputSize));
            CUDA_CHECK(cudaMalloc(&m_inputMemory[1],m_inputSize));


            CUDA_CHECK(cudaMallocHost(&m_outputMemory[0],m_outputSize));
            CUDA_CHECK(cudaMalloc(&m_outputMemory[1],m_outputSize));

            m_bindings[0]=m_inputMemory[1];
            m_bindings[1]= m_outputMemory[1];

        }


void Classifier::reset_task(){};//虽然没有用但是也要去写一下

        



bool Classifier::preprocess_cpu() {
    /*Preprocess -- 获取mean, std*/
    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};

    cv::Mat input_image;
    input_image = cv::imread(m_imagePath);
    if (input_image.data == nullptr) {
        LOGE("ERROR: Image file not founded! Program terminated"); 
        return false;
    }

    /*Preprocess -- 测速*/
    m_timer->start_cpu();


    cv::resize(input_image, input_image, 
               cv::Size(m_params.img.w, m_params.img.h), 0, 0, cv::INTER_LINEAR);

    /*Preprocess -- host端进行normalization和BGR2RGB, NHWC->NCHW*/
    int index;
    int offset_ch0 = m_imgArea * 0;
    int offset_ch1 = m_imgArea * 1;
    int offset_ch2 = m_imgArea * 2;
    for (int i = 0; i < m_inputDims.d[2]; i++) {
        for (int j = 0; j < m_inputDims.d[3]; j++) {
            index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
            m_inputMemory[0][offset_ch2++] = (input_image.data[index + 0] / 255.0f - mean[0]) / std[0];
            m_inputMemory[0][offset_ch1++] = (input_image.data[index + 1] / 255.0f - mean[1]) / std[1];
            m_inputMemory[0][offset_ch0++] = (input_image.data[index + 2] / 255.0f - mean[2]) / std[2];


          }
        }


    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

    m_timer->stop_cpu<timer::Timer::ms>("preprocess(CPU)");

    return true;
}


bool Classifier::preprocess_gpu()
{

     float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};


    cv::Mat input_image;

    input_image = cv::imread(m_imagePath);
    if(input_image.data==nullptr)
    {
        LOGD("not found %s",m_imagePath.c_str());
        return false;
    }

    m_timer->start_gpu();
    preprocess::preprocess_resize_gpu(input_image,m_inputMemory[1],m_params.img.h,m_params.img.w,preprocess::tactics::GPU_BILINEAR_CENTER);
    m_timer->stop_gpu("preprocess_gpu");

    //直接在gpu中然后进行推理

    return true;
}

bool Classifier::postprocess_cpu()
{


    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};

    m_timer->start_cpu();


    CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0],m_outputMemory[1],m_outputSize,cudaMemcpyKind::cudaMemcpyDeviceToHost,m_stream));

    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    //计算置信度

    ImageNetLabels labels;

    int pos =max_element(m_outputMemory[0],m_outputMemory[0]+m_params.num_cls)-m_outputMemory[0];  //算出来位置

    m_confidence= m_outputMemory[0][pos]*100;
    m_timer->stop_cpu<timer::Timer::ms>("postcpu");



    
    LOG("Result:     %s", labels.imagenet_labelstring(pos).c_str());   
    LOG("Confidence  %.3f%%", m_confidence);   
    m_timer->show();
    printf("\n");

    return true;

}

bool Classifier::postprocess_gpu()
{
    postprocess_cpu();
    return true;
}


std::shared_ptr<Classifier>make_classifier(std::string onnx_path,logger::Level level,Params params)
{

    return std::make_shared<Classifier> (onnx_path,level,params);

    //返回一个指向classifier的指针并初始化

}

    }//classifier

}//model






