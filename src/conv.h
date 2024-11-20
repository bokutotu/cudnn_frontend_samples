#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <cudnn.h>
#include "cudnn_frontend_wrapper.h"

typedef void* ConvGraph_t;
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
    const CudnnTensorDescriptor_t* input_desc,
    const CudnnTensorDescriptor_t* filter_desc,
    const CudnnTensorDescriptor_t* output_desc,
    const ConvConvolutionDescriptor_t* conv_desc,
    CudnnFrontendDataType_t data_type);

CudnnFrontendError_t build_dgrad_graph(
    cudnnHandle_t handle,
    ConvGraph_t* graph_out,
    const CudnnTensorDescriptor_t* dy_desc,
    const CudnnTensorDescriptor_t* w_desc,
    const CudnnTensorDescriptor_t* dx_desc,
    const ConvConvolutionDescriptor_t* conv_desc,
    CudnnFrontendDataType_t data_type);

CudnnFrontendError_t build_wgrad_graph(
    cudnnHandle_t handle,
    ConvGraph_t* graph_out,
    const CudnnTensorDescriptor_t* x_desc,
    const CudnnTensorDescriptor_t* dy_desc,
    const CudnnTensorDescriptor_t* dw_desc,
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

