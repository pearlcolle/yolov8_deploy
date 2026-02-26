#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc//imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "trt_detector.hpp"
#include "trt_preprocess.hpp"
#include "coco_labels.hpp"




using namespace std;
using namespace nvinfer1;


namespace model{

    namespace detector{


        //计算iou
        float iou_calc(bbox bbox1, bbox bbox2)
        {

            auto interx0 = std::max(bbox1.x0,bbox2.x0);
            auto intery0 = std::max(bbox1.y0,bbox2.y0);

            auto interx1 = std::min(bbox1.x1,bbox2.x1);
            auto intery1 = std::min(bbox1.y1,bbox2.y1);

            float interw = interx1-interx0;
            float interh = intery1-intery0;

            float inter = interw*interh;

            float unionn = (bbox1.x1-bbox1.x0)*(bbox1.y1-bbox1.y0)+
                           (bbox2.x1-bbox2.x0)*(bbox2.y1-bbox2.y0)
                           -inter;
            return inter/unionn;
        }


        void Detector::setup(void const* data, std::size_t size)
        {

            //setup是用来生成engine反序列化，分配空间的作用

            m_runtime = shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_logger));
            m_engine = shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(data,size));
            m_context = shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());


            m_inputDims = m_context->getBindingDimensions(0);
            m_outputDims = m_context->getBindingDimensions(1);  //获取两个输入输出的维度


             CUDA_CHECK(cudaStreamCreate(&m_stream));//创建流

            //计算输入输出的size来分配内存

            m_inputSize  = m_params.img.w*m_params.img.h*m_params.img.c*sizeof(float);

            m_outputSize = m_outputDims.d[1]*m_outputDims.d[2]*sizeof(float);
            m_imgArea =  m_params.img.w*m_params.img.h;



            //给CPU分配内存
            CUDA_CHECK(cudaMallocHost(&m_inputMemory[0],m_inputSize));
            CUDA_CHECK(cudaMallocHost(&m_outputMemory[0],m_outputSize));

            //GPU
            CUDA_CHECK(cudaMalloc(&m_inputMemory[1],m_inputSize));
            CUDA_CHECK(cudaMalloc(&m_outputMemory[1],m_outputSize));


            m_bindings[0]= m_inputMemory[1];
            m_bindings[1]= m_outputMemory[1]; //注意，这一步是让 m_bindings[0]与m_inputMemory[1]指向同一个物理位置,之后再寻址就直接从这里找
        }


    void Detector::reset_task(){
          m_bboxes.clear();
                   }

    

    bool Detector::preprocess_cpu()
    {

        cv::Mat image = cv::imread(m_imagePath);
        if(image.data==nullptr)
        {
            LOGE("load fail_image");
            return false;
        };

        m_timer->start_cpu();
        //进行等比例缩放
        float input_w =image.cols;
        float input_h =image.rows;
        
        
        float tar_w = m_params.img.w;
        float tar_h =m_params.img.h; //fang

        float scale_w = tar_w/input_w;
        float scale_h = tar_h/input_h;

        float scale =min(scale_w,scale_h);

        int new_w = input_w*scale;
        int new_h = input_h*scale; //等比例缩放后的图片大小，但不是放入模型的大小
        cv::Mat  resize_image;

        cv::resize(image,resize_image,cv::Size(new_w,new_h));


        preprocess::warpaffine_init(input_h, input_w, tar_h, tar_w);

        cv::Mat tar(tar_h,tar_w,CV_8UC3,cv::Scalar(0,0,0));//创建一个黑纸，把缩放好的图片放上去，然后给模型

        //算偏移量
        int x,y;
        x = (new_w<tar_w)?(tar_w-new_w)/2:0;
        y =(new_h<tar_h)?(tar_h-new_h)/2:0;

        //创建一个矩形，这个矩形就是缩放后的图片放到tar的位置

        cv::Rect roi(x,y,new_w,new_h);

        cv::Mat roii = tar(roi);

        resize_image.copyTo(roii);


        //现在tar就是一张包含resize_image其余都是黑边的图片,现在开始遍历传给m_inputmemory中去
        //
        int cha0 = m_imgArea*0;
        int cha1 = m_imgArea*1;
        int cha2 = m_imgArea*2;

        for(int i=0;i<tar_h;i++)
        {
            for(int j=0;j<tar_w;j++)
            {
                int idx = (i*tar_w+j)*3; //由于是3通道所以要×3
                m_inputMemory[0][cha0++]=tar.data[idx+2]/255.0f;
                m_inputMemory[0][cha1++]=tar.data[idx+1]/255.0f;
                m_inputMemory[0][cha2++]=tar.data[idx+0]/255.0f;
            }
        }

        CUDA_CHECK( cudaMemcpyAsync(m_inputMemory[1],m_inputMemory[0],m_inputSize,cudaMemcpyKind::cudaMemcpyHostToDevice,m_stream));
        m_timer->stop_cpu<timer::Timer::ms>("preprocess(CPU)");
        return true;

    }



    bool Detector::preprocess_gpu() {

    /*Preprocess -- 读取数据*/
    m_inputImage = cv::imread(m_imagePath);
    if (m_inputImage.data == nullptr) {
        LOGE("ERROR: file not founded! Program terminated"); return false;
    }
    
    /*Preprocess -- 测速*/
    m_timer->start_gpu();

    /*Preprocess -- 使用GPU进行warpAffine, 并将结果返回到m_inputMemory中*/
    preprocess::preprocess_resize_gpu(m_inputImage, m_inputMemory[1],
                                   m_params.img.h, m_params.img.w, 
                                   preprocess::tactics::GPU_WARP_AFFINE);
    m_timer->stop_gpu("preprocess(GPU)");
    return true;
}


    























































































    } //detector

} //model