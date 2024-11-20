#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <cudnn.h>

typedef void* ConvGraph_t;

typedef enum {
    CONV_SUCCESS = 0,
    CONV_FAILURE = 1,
    CONV_INVALID_VALUE = 2,
    CONV_NOT_SUPPORTED = 3,
    // Add more error codes as needed
} CudnnFrontendError_t;

typedef enum {
    CONV_DATA_TYPE_HALF,
    CONV_DATA_TYPE_FLOAT,
    CONV_DATA_TYPE_DOUBLE,
    // Add more as needed
} CudnnFrontendDataType_t;

typedef struct {
    size_t num_dims;
    int64_t dims[8];    // Maximum of 8 dimensions
    int64_t strides[8]; // Corresponding strides
} ConvTensorDescriptor_t;

typedef struct {
    size_t num_dims;
    int64_t padding[8];   // Padding for each spatial dimension
    int64_t stride[8];    // Stride for each spatial dimension
    int64_t dilation[8];  // Dilation for each spatial dimension
} ConvConvolutionDescriptor_t;

// Build graphs
CudnnFrontendError_t build_fprop_graph(
    cudnnHandle_t handle,
    ConvGraph_t* graph_out,
    const ConvTensorDescriptor_t* input_desc,
    const ConvTensorDescriptor_t* filter_desc,
    const ConvTensorDescriptor_t* output_desc,
    const ConvConvolutionDescriptor_t* conv_desc,
    CudnnFrontendDataType_t data_type);

CudnnFrontendError_t build_dgrad_graph(
    cudnnHandle_t handle,
    ConvGraph_t* graph_out,
    const ConvTensorDescriptor_t* dy_desc,
    const ConvTensorDescriptor_t* w_desc,
    const ConvTensorDescriptor_t* dx_desc,
    const ConvConvolutionDescriptor_t* conv_desc,
    CudnnFrontendDataType_t data_type);

CudnnFrontendError_t build_wgrad_graph(
    cudnnHandle_t handle,
    ConvGraph_t* graph_out,
    const ConvTensorDescriptor_t* x_desc,
    const ConvTensorDescriptor_t* dy_desc,
    const ConvTensorDescriptor_t* dw_desc,
    const ConvConvolutionDescriptor_t* conv_desc,
    CudnnFrontendDataType_t data_type);

// Get workspace size
CudnnFrontendError_t get_workspace_size(ConvGraph_t graph, size_t* workspace_size);

// Execute graph
CudnnFrontendError_t execute_graph(
    cudnnHandle_t handle,
    ConvGraph_t graph,
    void* input_ptrs[],
    void* output_ptrs[],
    void* workspace);

// Destroy graph
void destroy_graph(ConvGraph_t graph);

// Get number of inputs and outputs
CudnnFrontendError_t get_num_inputs(ConvGraph_t graph, size_t* num_inputs);
CudnnFrontendError_t get_num_outputs(ConvGraph_t graph, size_t* num_outputs);

#ifdef __cplusplus
}
#endif

// #endif // CONV_API_H

