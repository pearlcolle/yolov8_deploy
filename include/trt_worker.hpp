#ifndef __WORKER_HPP__
#define __WORKER_HPP__

#include <memory>
#include <vector>
#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "trt_classifier.hpp"
#include "trt_detector.hpp"



namespace thread{

    class Worker
    {
        public:

        Worker(std::string onnxPath, logger::Level level, model::Params params);
        void do_inference(std::string imagePath);

        public:

         std::shared_ptr<logger::Logger>          m_logger;
         std::shared_ptr<model::Params>           m_params;


        std::shared_ptr<model::classifier::Classifier>  m_classifier;//创建的成员对象去接受classifier中make_classifier返回的对象
        std::shared_ptr<model::detector::Detector>      m_detector;
        std::vector<float>                              m_scores;
        std::vector<model::detector::bbox>              m_boxes;


        };

std::shared_ptr<Worker> create_worker(std::string onnxPath, logger::Level level, model::Params params);


}

#endif //__WORKER_HPP__