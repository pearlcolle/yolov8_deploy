#ifndef __TRT_CLASSIFIER_HPP__
#define __TRT_CLASSIFIER_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "trt_logger.hpp"
#include "trt_model.hpp"


//在model作用域下在创建一个classfier的作用域，储存classifer的类来实现classifier的一些特殊功能
namespace model{

    namespace classifier{


        class Classifier : public Model //继承Model父类的
        {
            public:
            Classifier(std::string onnx_path,logger::Level level,Params params):
                Model(onnx_path,level,params)
                {};//调用的其实是父类的构造函数

            public:

            //这些函数都是需要在classifier.cpp中去实现细节的
                virtual void setup(void const* data, std::size_t size)override;
                virtual void reset_task() override;
                virtual bool preprocess_cpu()override;
                virtual bool preprocess_gpu()override;
                virtual bool postprocess_cpu()override;
                virtual bool postprocess_gpu()override;

            private:
            float m_confidence;//置信度
            std::string m_label; //标签

            int m_inputSize; 
            int m_imgArea;
            int m_outputSize;


        };
    std:: shared_ptr<Classifier>make_classifier(std::string onnx_path,logger::Level level,Params params);

    };
}

































#endif //__TRT_CLASSIFIER_HPP__