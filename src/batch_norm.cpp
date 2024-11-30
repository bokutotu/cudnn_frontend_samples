#include "batch_norm.h"
#include "cudnn_frontend_wrapper.h"
#include "utils.h"

#include <cudnn.h>
#include <cudnn_frontend.h>
#include <iostream>
#include <string>

struct BNStruct_t {
    cudnn_frontend::graph::Graph* graph_ptr;
};

CudnnTensorDescriptor_t getSumDesc(const CudnnTensorDescriptor_t* input) {
    auto len_dims = input->num_dims;

    if (len_dims == 4) {
        CudnnTensorDescriptor_t sum_desc;
        sum_desc.num_dims = 4;
        sum_desc.dims[0] = 1;
        sum_desc.dims[1] = input->dims[1];
        sum_desc.dims[2] = 1;
        sum_desc.dims[3] = 1;
        sum_desc.strides[0] = input->dims[1];
        sum_desc.strides[1] = 1;
        sum_desc.strides[2] = input->dims[1];
        sum_desc.strides[3] = input->dims[1];
        return sum_desc;
    } else if (len_dims == 2) {
        CudnnTensorDescriptor_t sum_desc;
        sum_desc.num_dims = 2;
        sum_desc.dims[0] = 1;
        sum_desc.dims[1] = input->dims[1];
        sum_desc.strides[0] = input->strides[0];
        sum_desc.strides[1] = input->strides[1];
        return sum_desc;
    } else {
        return nullptr;
    }
}

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
    int64_t accum_count,
    int mode,  // 0: inference, 1: training
    CudnnFrontendDataType_t data_type
) {
    if (graph_out == nullptr || x_desc == nullptr || scale_desc == nullptr ||
        bias_desc == nullptr || mean_desc == nullptr || var_desc == nullptr ||
        y_desc == nullptr) {
        return INVALID_VALUE;
    }

    // namesapce fe = cudnn_frontend;
    cudnn_frontend::graph::Graph graph;
    cudnn_frontend::DataType_t cudnn_data_type = getCudnnDataType(data_type);
    graph.set_io_data_type(cudnn_data_type)
        .set_intermediate_data_type(cudnn_data_type)
        .set_compute_data_type(cudnn_data_type);

    if (mode == 1) {
        // auto sum_desc = getTensorDescriptor(sum);
        auto sum_shape = getSumDesc(x_desc);
        auto sum = graph.tensor(getTensorAttributes(&sum_shape, "sum"));
        auto sq_sum = graph.tensor(getTensorAttributes(&sum_shape, "sq_sum"));
        auto prev_running_mean = graph.tensor(getTensorAttributes(mean_desc, "prev_running_mean"));
        auto prev_running_var = graph.tensor(getTensorAttributes(var_desc, "prev_running_var"));
        auto scale = graph.tensor(getTensorAttributes(scale_desc, "scale"));
        auto bias = graph.tensor(getTensorAttributes(bias_desc, "bias"));

        auto epsilon_tensor = graph.tensor(epsilon);
        auto momentum_tensor = graph.tensor(momentum);
        auto accum_count_tensor = graph.tensor(accum_count);

        auto bn_finalize_options = cudnn_frontend::graph::BN_finalize_attributes()
            .set_previous_running_stats(prev_running_mean, prev_running_var, momentum_tensor);
        auto [eq_scale, eq_bias, saved_mean, saved_inv_variance, next_running_mean, next_running_var] =
            graph.bn_finalize(sum, sq_sum, scale, bias, epsilon_tensor, accum_count_tensor, bn_finalize_options);
        eq_scale->set_output(true);
        eq_bias->set_output(true);
        saved_mean->set_output(true);
        saved_inv_variance->set_output(true);
        next_running_mean->set_output(true);
        next_running_var->set_output(true);
    } else {
        //  todo
    }

    cudnn_frontend::error_t status = graph.validate();
    if (!status.is_good()) {
        return FAILURE;
    }

    status = graph.build_operation_graph(handle);
    if (!status.is_good()) {
        return FAILURE;
    }

    status = graph.create_execution_plans({cudnn_frontend::HeurMode_t::FALLBACK});
    if (!status.is_good()) {
        return FAILURE;
    }

    BNStruct_t* bn_struct = new BNStruct_t();
    bn_struct->graph_ptr = &graph;

    return SUCCESS;
}

// CudnnFrontendError_t build_bn_backward_graph(
//     cudnnHandle_t handle,
//     BNGraph_t* graph_out,
//     const CudnnTensorDescriptor_t* x_desc,      // Input tensor descriptor
//     const CudnnTensorDescriptor_t* dy_desc,     // Gradient output tensor descriptor
//     const CudnnTensorDescriptor_t* scale_desc,  // Scale tensor descriptor
//     const CudnnTensorDescriptor_t* saved_mean_desc,       // Saved mean tensor descriptor
//     const CudnnTensorDescriptor_t* saved_inv_var_desc,    // Saved inverse variance tensor descriptor
//     const CudnnTensorDescriptor_t* dx_desc,     // Gradient input tensor descriptor (output)
//     const CudnnTensorDescriptor_t* dscale_desc, // Gradient scale tensor descriptor (output)
//     const CudnnTensorDescriptor_t* dbias_desc,  // Gradient bias tensor descriptor (output)
//     int mode,  // 0: per-activation, 1: spatial
//     CudnnFrontendDataType_t data_type
// ) {
//     return SUCCESS;
// }
