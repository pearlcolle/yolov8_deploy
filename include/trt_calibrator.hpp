#ifndef __TRT_CALIBRATOR_HPP__
#define __TRT_CALIBRATOR_HPP__
#include "NvInfer.h"
#include <string>
#include <vector>


namespace model
{




    class Int8EntropyCalibrator: public nvinfer1::IInt8EntropyCalibrator2{


        public:
        Int8EntropyCalibrator();


        ~Int8EntropyCalibrator();


        int32_t getBatchSize() const noexcept override{return m_batchSize};

        bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept override;

        































    }








































    
} 
































#endif __TRT_CALIBRATOR_HPP__