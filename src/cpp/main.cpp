#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "trt_worker.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    
    string onnxPath    = "/home/ubuntu24/yolov8/models/onnx/yolov8n.onnx";

    auto level         = logger::Level::VERB;
    auto params        = model::Params();

    params.img         = {640, 640, 3};
    params.tas       = model::task_type::DETECTION;
    params.dev         = model::device::GPU;
    params.pre        = model::precision::FP16;

    // 创建一个worker的实例, 在创建的时候就完成初始化
    auto worker   = thread::create_worker(onnxPath, level, params);

    // 根据worker中的task类型进行推理
    worker->do_inference("data/source/car.jpg");
    worker->do_inference("data/source/crowd.jpg");
    worker->do_inference("data/source/crossroad.jpg");
    worker->do_inference("data/source/airport.jpg");
    worker->do_inference("data/source/bedroom.jpg");
    worker->do_inference("data/source/aaa.jpg");
    worker->do_inference("data/source/bbb.jpg");
    worker->do_inference("data/source/ccc.jpg");

    return 0;
}