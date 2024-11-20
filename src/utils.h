#pragma once

#include <cudnn.h>
#include <cudnn_frontend.h>
#include "cudnn_frontend_wrapper.h"

static cudnn_frontend::DataType_t getCudnnDataType(CudnnFrontendDataType_t data_type) {
    using namespace cudnn_frontend;
    switch (data_type) {
        case DATA_TYPE_HALF:
            return DataType_t::HALF;
        case DATA_TYPE_FLOAT:
            return DataType_t::FLOAT;
        case DATA_TYPE_DOUBLE:
            return DataType_t::DOUBLE;
        // Add more cases as needed
        default:
            return DataType_t::FLOAT; // Default to float
    }
}
