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

            float interw = std::max( 0.0f,interx1-interx0);
            float interh = std::max( 0.0f,intery1-intery0); //防止为负

            float inter = interw*interh;

            float unionn = (bbox1.x1-bbox1.x0)*(bbox1.y1-bbox1.y0)+
                           (bbox2.x1-bbox2.x0)*(bbox2.y1-bbox2.y0)
                           -inter;
            return inter/unionn;
        }


        void Detector::setup(void const* data, std::size_t size)
        {

            //setup是用来生成engine反序列化，分配空间的作用

            m_runtime = shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_logger),destroy_ptr<IRuntime>);
            m_engine = shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(data,size),destroy_ptr<ICudaEngine>);
            m_context = shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext(),destroy_ptr<IExecutionContext>);


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
    
    m_timer->start_gpu();

    preprocess::preprocess_resize_gpu(m_inputImage, m_inputMemory[1],
                                   m_params.img.h, m_params.img.w, 
                                   preprocess::tactics::GPU_WARP_AFFINE);
    m_timer->stop_gpu("preprocess(GPU)");
    return true;
}

bool Detector::postprocess_cpu()
{


    m_timer->start_cpu();

    //推理完了，我要把数据放回cpu中进行后处理
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0],m_outputMemory[1],m_outputSize,cudaMemcpyKind::cudaMemcpyDeviceToHost,m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));



      /*Postprocess -- 1. decode*/
    /*
     * 我们需要做的就是将[batch, bboxes, ch]转换为vector<bbox>
     * 几个步骤:
     * 1. 从每一个bbox中对应的ch中获取cx, cy, width, height
     * 2. 对每一个bbox中对应的ch中，找到最大的class label, 可以使用std::max_element
     * 3. 将cx, cy, width, height转换为x0, y0, x1, y1
     * 4. 因为图像是经过resize了的，所以需要根据resize的scale和shift进行坐标的转换(这里面可以根据preprocess中的到的affine matrix来进行逆变换)
     * 5. 将转换好的x0, y0, x1, y1，以及confidence和classness给存入到box中，并push到m_bboxes中，准备接下来的NMS处理
     */
    //outputdim(1,8400,84)

    float conf_threshold = 0.25; //用来过滤decode时的bboxes
    float nms_threshold  = 0.45;  //用来过滤nms时的bboxes


    int    boxes_count = m_outputDims.d[1];//8400
    int    class_count = m_outputDims.d[2] - 4;//84-40
    float* tensor;

    float  cx, cy, w, h, obj, prob, conf;  //中心点，宽高，物体（0，1），概率，置信度
    float  x0, y0, x1, y1; //框的4个点
    int    label; //标签索引


    //遍历每个8400个框去解码
    for(int i=0;i<boxes_count;i++)
    {
        tensor = m_outputMemory[0]+i*m_outputDims.d[2];//每次循环都是下一个框
        label = max_element(tensor+4,tensor+4+class_count)-(tensor+4);

        conf =tensor[label+4];

        if(conf<conf_threshold)
        {
            continue;
        }
        cx =tensor[0];
        cy =tensor[1];
        w = tensor[2];
        h = tensor[3];

        x0 = cx-w/2;
        y0 =cy-h/2;
        x1 = x0+w;
        y1 = y0+h;


        //由于图片已经resize了，我们需要把框的坐标还原到原图上，所以要进行affine_matrix;

        preprocess::affine_transformation(preprocess::affine_matrix.reverse,x0,y0,&x0,&y0);
        preprocess::affine_transformation(preprocess::affine_matrix.reverse,x1,y1,&x1,&y1);
        //此时框的坐标已经确定，现在给他放到bbox的vecrtor中

        bbox bboxx(x0,y0,x1,y1,conf,label);
        m_bboxes.emplace_back(bboxx);
    }

      LOGD("the count of decoded bbox is %d", m_bboxes.size()); //打印出第一次过滤后的box的数量



      //先对m_box进行排列，根据置信度

      std::vector<bbox> final_bbox;
      final_bbox.reserve(m_bboxes.size());
      sort(m_bboxes.begin(),m_bboxes.end(),[](bbox &bbox1,bbox &bbox2){return bbox1.confidence>bbox2.confidence;});


      //第一个最大的框
      for(int i=0;i<m_bboxes.size();i++)
      {
        if(m_bboxes[i].flg_remove)
        {
            continue;
        }
        final_bbox.emplace_back(m_bboxes[i]);

        //第二个最大的框
        for(int j=i+1;j<m_bboxes.size();j++)
        {
            if(m_bboxes[j].flg_remove)
            {
                continue;
            }


            float iou = iou_calc(m_bboxes[i],m_bboxes[j]);
            if(iou>nms_threshold)
            {
                m_bboxes[j].flg_remove=true;
            }

        }
      }
      LOGD("the count of bbox after NMS is %d", final_bbox.size());



      //draw boxes

      CocoLabels labels;
      for (int i = 0; i < final_bbox.size(); i++) {
          auto box = final_bbox[i];

          string name = labels.coco_get_label(box.label);
          string txt = cv::format("%s: %.2f", name.c_str(), box.confidence);

           cv::Point pt1(box.x0, box.y0); // 左上角
           cv::Point pt2(box.x1, box.y1); // 右下角

   
           cv::rectangle(m_inputImage, pt1, pt2, cv::Scalar(0, 255, 0), 2);

           cv::Point txt_pos(box.x0, box.y0 - 5); 
           cv::putText(m_inputImage, txt, txt_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        
        }
        static int image_counter = 0; 
        std::string save_path = "/home/ubuntu24/yolov8/data/results/result_" + std::to_string(image_counter++) + ".png";
    
        cv::imwrite(save_path, m_inputImage);
        m_timer->stop_cpu<timer::Timer::ms>("postprocess(CPU)"); 
        m_timer->show();                              
        printf("\n");

        return true;

}

bool Detector::postprocess_gpu()
{
    m_timer->start_gpu();
    Detector::postprocess_cpu();
    m_timer->stop_gpu();

    return true;
}




std::shared_ptr<Detector> make_detector(std::string onnx_path, logger::Level level, Params params)
{
    return make_shared<Detector>(onnx_path, level, params);
}

    } //detector

} //model