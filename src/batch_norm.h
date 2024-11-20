#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <cudnn.h>
#include "cudnn_frontend_wrapper.h"
#include "utils.h"

typedef void* BNGraph_t;

// Build BN forward graph
CudnnFrontendError_t build_bn_forward_graph(
    cudnnHandle_t handle,
    BNGraph_t* graph_out,
    const CudnnTensorDescriptor_t* x_desc,      // Input tensor descriptor
    const CudnnTensorDescriptor_t* scale_desc,  // Scale tensor descriptor
    const CudnnTensorDescriptor_t* bias_desc,   // Bias tensor descriptor
    const CudnnTensorDescriptor_t* mean_desc,   // Running mean tensor descriptor (for inference)
    const CudnnTensorDescriptor_t* var_desc,    // Running variance tensor descriptor (for inference)
    const CudnnTensorDescriptor_t* y_desc,      // Output tensor descriptor
    float epsilon,
    float momentum,
    int mode  // 0: inference, 1: training
);

// Build BN backward graph
CudnnFrontendError_t build_bn_backward_graph(
    cudnnHandle_t handle,
    BNGraph_t* graph_out,
    const CudnnTensorDescriptor_t* x_desc,      // Input tensor descriptor
    const CudnnTensorDescriptor_t* dy_desc,     // Gradient output tensor descriptor
    const CudnnTensorDescriptor_t* scale_desc,  // Scale tensor descriptor
    const CudnnTensorDescriptor_t* saved_mean_desc,       // Saved mean tensor descriptor
    const CudnnTensorDescriptor_t* saved_inv_var_desc,    // Saved inverse variance tensor descriptor
    const CudnnTensorDescriptor_t* dx_desc,     // Gradient input tensor descriptor (output)
    const CudnnTensorDescriptor_t* dscale_desc, // Gradient scale tensor descriptor (output)
    const CudnnTensorDescriptor_t* dbias_desc,  // Gradient bias tensor descriptor (output)
    int mode  // 0: per-activation, 1: spatial
);

// Get workspace size
CudnnFrontendError_t get_bn_workspace_size(BNGraph_t graph, size_t* workspace_size);

// Execute BN graph
CudnnFrontendError_t execute_bn_graph(
    cudnnHandle_t handle,
    BNGraph_t graph,
    void* input_ptrs[],
    void* output_ptrs[],
    void* workspace);

// Destroy BN graph
void destroy_bn_graph(BNGraph_t graph);

// Get number of inputs and outputs
CudnnFrontendError_t get_bn_num_inputs(BNGraph_t graph, size_t* num_inputs);
CudnnFrontendError_t get_bn_num_outputs(BNGraph_t graph, size_t* num_outputs);

#ifdef __cplusplus
}
#endif
