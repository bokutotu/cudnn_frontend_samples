#pragma once

#include <cudnn.h>
#include <cudnn_frontend.h>
#include "cudnn_frontend_wrapper.h"
#include <string>

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

cudnn_frontend::TensorDescriptor_t getTensorDescriptor(const CudnnTensorDescriptor_t* desc) {
    using namespace cudnn_frontend;
    std::vector<int64_t> dims(desc->dims, desc->dims + desc->num_dims);
    std::vector<int64_t> strides(desc->strides, desc->strides + desc->num_dims);
    return Tensor_attributes().set_dim(dims).set_stride(strides);
}

cudnn_frontend::graph::Tensor_attributes getTensorAttributes(const CudnnTensorDescriptor_t* desc, std::string name) {
    using namespace cudnn_frontend;
    std::vector<int64_t> dims(desc->dims, desc->dims + desc->num_dims);
    std::vector<int64_t> strides(desc->strides, desc->strides + desc->num_dims);
    return cudnn_frontend::graph::Tensor_attributes().set_dim(dims).set_stride(strides).set_name(name);
}
