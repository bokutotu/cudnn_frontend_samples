#include "batch_norm.h"

#include <cudnn_frontend.h>
#include <memory>
#include <vector>
#include <unordered_map>

struct BNGraph {
    std::shared_ptr<cudnn_frontend::graph::Graph> graph_ptr;
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> input_tensors;
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> output_tensors;
    float epsilon;
    float momentum;
    int mode;  // 0: inference, 1: training
};
static std::vector<int64_t> vector_from_array(const int64_t* array, size_t size) {
    return std::vector<int64_t>(array, array + size);
}

extern "C" {

CudnnFrontendError_t build_bn_forward_graph(
    cudnnHandle_t handle,
    BNGraph_t* graph_out,
    const CudnnTensorDescriptor_t* x_desc,
    const CudnnTensorDescriptor_t* scale_desc,
    const CudnnTensorDescriptor_t* bias_desc,
    const CudnnTensorDescriptor_t* mean_desc,   // For inference
    const CudnnTensorDescriptor_t* var_desc,    // For inference
    const CudnnTensorDescriptor_t* y_desc,
    float epsilon,
    float momentum,
    int mode  // 0: inference, 1: training
) {
    if (graph_out == nullptr || x_desc == nullptr || scale_desc == nullptr ||
        bias_desc == nullptr || y_desc == nullptr) {
        return INVALID_VALUE;
    }
    try {
        using namespace cudnn_frontend;
        BNGraph* bn_graph = new BNGraph();

        auto graph = std::make_shared<graph::Graph>();

        auto data_type = getCudnnDataType(x_desc->data_type);

        graph->set_io_data_type(data_type)
            .set_intermediate_data_type(DataType_t::FLOAT)
            .set_compute_data_type(DataType_t::FLOAT);

        // Create tensors
        auto X = graph->tensor(graph::Tensor_attributes()
                                   .set_name("X")
                                   .set_dim(vector_from_array(x_desc->dims, x_desc->num_dims))
                                   .set_stride(vector_from_array(x_desc->strides, x_desc->num_dims)));

        auto Scale = graph->tensor(graph::Tensor_attributes()
                                       .set_name("scale")
                                       .set_dim(vector_from_array(scale_desc->dims, scale_desc->num_dims))
                                       .set_stride(vector_from_array(scale_desc->strides, scale_desc->num_dims))
                                       .set_data_type(DataType_t::FLOAT));

        auto Bias = graph->tensor(graph::Tensor_attributes()
                                      .set_name("bias")
                                      .set_dim(vector_from_array(bias_desc->dims, bias_desc->num_dims))
                                      .set_stride(vector_from_array(bias_desc->strides, bias_desc->num_dims))
                                      .set_data_type(DataType_t::FLOAT));

        auto epsilon_tensor = graph->tensor(epsilon);
        auto momentum_tensor = graph->tensor(momentum);

        graph::Batchnorm_attributes batchnorm_options;
        batchnorm_options.set_epsilon(epsilon_tensor);

        std::shared_ptr<graph::Tensor_attributes> Mean, InvVar;
        std::shared_ptr<graph::Tensor_attributes> NextRunningMean, NextRunningVar;

        if (mode == 0) {
            // Inference mode
            auto Mean_in = graph->tensor(graph::Tensor_attributes()
                                             .set_name("running_mean")
                                             .set_dim(vector_from_array(mean_desc->dims, mean_desc->num_dims))
                                             .set_stride(vector_from_array(mean_desc->strides, mean_desc->num_dims))
                                             .set_data_type(DataType_t::FLOAT));

            auto Var_in = graph->tensor(graph::Tensor_attributes()
                                            .set_name("running_var")
                                            .set_dim(vector_from_array(var_desc->dims, var_desc->num_dims))
                                            .set_stride(vector_from_array(var_desc->strides, var_desc->num_dims))
                                            .set_data_type(DataType_t::FLOAT));

            batchnorm_options.set_running_stats(Mean_in, Var_in);
            // Build BN inference
            auto Y = graph->batchnorm_inference(X, Mean_in, Var_in, Scale, Bias, batchnorm_options);
            Y->set_dim(vector_from_array(y_desc->dims, y_desc->num_dims)).set_output(true);

            // Set inputs and outputs
            bn_graph->input_tensors = {X, Scale, Bias, Mean_in, Var_in};
            bn_graph->output_tensors = {Y};
        } else {
            // Training mode
            auto PrevRunningMean = graph->tensor(graph::Tensor_attributes()
                                                     .set_name("prev_running_mean")
                                                     .set_dim(vector_from_array(mean_desc->dims, mean_desc->num_dims))
                                                     .set_stride(vector_from_array(mean_desc->strides, mean_desc->num_dims))
                                                     .set_data_type(DataType_t::FLOAT));

            auto PrevRunningVar = graph->tensor(graph::Tensor_attributes()
                                                    .set_name("prev_running_var")
                                                    .set_dim(vector_from_array(var_desc->dims, var_desc->num_dims))
                                                    .set_stride(vector_from_array(var_desc->strides, var_desc->num_dims))
                                                    .set_data_type(DataType_t::FLOAT));

            batchnorm_options.set_previous_running_stats(PrevRunningMean, PrevRunningVar, momentum_tensor);

            // Build BN forward
            auto [Y, Mean, InvVar, NextRunningMean, NextRunningVar] =
                graph->batchnorm(X, Scale, Bias, batchnorm_options);

            Y->set_dim(vector_from_array(y_desc->dims, y_desc->num_dims)).set_output(true);
            Mean->set_output(true).set_data_type(DataType_t::FLOAT);
            InvVar->set_output(true).set_data_type(DataType_t::FLOAT);
            NextRunningMean->set_output(true).set_data_type(DataType_t::FLOAT);
            NextRunningVar->set_output(true).set_data_type(DataType_t::FLOAT);

            // Set inputs and outputs
            bn_graph->input_tensors = {X, Scale, Bias, PrevRunningMean, PrevRunningVar};
            bn_graph->output_tensors = {Y, Mean, InvVar, NextRunningMean, NextRunningVar};
        }

        // Validate and build the graph
        auto status = graph->validate();
        if (!status.is_good()) {
            delete bn_graph;
            return FAILURE;
        }

        status = graph->build_operation_graph(handle);
        if (!status.is_good()) {
            delete bn_graph;
            return FAILURE;
        }

        status = graph->create_execution_plans({HeurMode_t::FALLBACK});
        if (!status.is_good()) {
            delete bn_graph;
            return FAILURE;
        }

        status = graph->check_support(handle);
        if (!status.is_good()) {
            delete bn_graph;
            return FAILURE;
        }

        status = graph->build_plans(handle);
        if (!status.is_good()) {
            delete bn_graph;
            return FAILURE;
        }

        bn_graph->graph_ptr = graph;
        bn_graph->epsilon = epsilon;
        bn_graph->momentum = momentum;
        bn_graph->mode = mode;

        *graph_out = static_cast<BNGraph_t>(bn_graph);
        return SUCCESS;

    } catch (const std::exception& e) {
        return FAILURE;
    }
}

// Build BN backward graph
CudnnFrontendError_t build_bn_backward_graph(
    cudnnHandle_t handle,
    BNGraph_t* graph_out,
    const CudnnTensorDescriptor_t* x_desc,
    const CudnnTensorDescriptor_t* dy_desc,
    const CudnnTensorDescriptor_t* scale_desc,
    const CudnnTensorDescriptor_t* saved_mean_desc,
    const CudnnTensorDescriptor_t* saved_inv_var_desc,
    const CudnnTensorDescriptor_t* dx_desc,
    const CudnnTensorDescriptor_t* dscale_desc,
    const CudnnTensorDescriptor_t* dbias_desc,
    int mode  // 0: per-activation, 1: spatial
) {
    if (graph_out == nullptr || x_desc == nullptr || dy_desc == nullptr ||
        scale_desc == nullptr || saved_mean_desc == nullptr || saved_inv_var_desc == nullptr ||
        dx_desc == nullptr || dscale_desc == nullptr || dbias_desc == nullptr) {
        return INVALID_VALUE;
    }
    try {
        using namespace cudnn_frontend;
        BNGraph* bn_graph = new BNGraph();

        auto graph = std::make_shared<graph::Graph>();

        auto data_type = getCudnnDataType(x_desc->data_type);

        graph->set_io_data_type(data_type)
            .set_intermediate_data_type(DataType_t::FLOAT)
            .set_compute_data_type(DataType_t::FLOAT);

        // Create tensors
        auto X = graph->tensor(graph::Tensor_attributes()
                                   .set_name("X")
                                   .set_dim(vector_from_array(x_desc->dims, x_desc->num_dims))
                                   .set_stride(vector_from_array(x_desc->strides, x_desc->num_dims)));

        auto DY = graph->tensor(graph::Tensor_attributes()
                                    .set_name("DY")
                                    .set_dim(vector_from_array(dy_desc->dims, dy_desc->num_dims))
                                    .set_stride(vector_from_array(dy_desc->strides, dy_desc->num_dims)));

        auto Scale = graph->tensor(graph::Tensor_attributes()
                                       .set_name("scale")
                                       .set_dim(vector_from_array(scale_desc->dims, scale_desc->num_dims))
                                       .set_stride(vector_from_array(scale_desc->strides, scale_desc->num_dims))
                                       .set_data_type(DataType_t::FLOAT));

        auto Mean = graph->tensor(graph::Tensor_attributes()
                                      .set_name("saved_mean")
                                      .set_dim(vector_from_array(saved_mean_desc->dims, saved_mean_desc->num_dims))
                                      .set_stride(vector_from_array(saved_mean_desc->strides, saved_mean_desc->num_dims))
                                      .set_data_type(DataType_t::FLOAT));

        auto InvVar = graph->tensor(graph::Tensor_attributes()
                                        .set_name("saved_inv_variance")
                                        .set_dim(vector_from_array(saved_inv_var_desc->dims, saved_inv_var_desc->num_dims))
                                        .set_stride(vector_from_array(saved_inv_var_desc->strides, saved_inv_var_desc->num_dims))
                                        .set_data_type(DataType_t::FLOAT));

        graph::Batchnorm_backward_attributes bn_backward_options;
        bn_backward_options.set_saved_mean_and_inv_variance(Mean, InvVar);

        // Build BN backward
        auto [DX, DScale, DBias] = graph->batchnorm_backward(DY, X, Scale, bn_backward_options);

        DX->set_dim(vector_from_array(dx_desc->dims, dx_desc->num_dims)).set_output(true);
        DScale->set_dim(vector_from_array(dscale_desc->dims, dscale_desc->num_dims)).set_output(true).set_data_type(DataType_t::FLOAT);
        DBias->set_dim(vector_from_array(dbias_desc->dims, dbias_desc->num_dims)).set_output(true).set_data_type(DataType_t::FLOAT);

        // Set inputs and outputs
        bn_graph->input_tensors = {DY, X, Scale, Mean, InvVar};
        bn_graph->output_tensors = {DX, DScale, DBias};

        // Validate and build the graph
        auto status = graph->validate();
        if (!status.is_good()) {
            delete bn_graph;
            return FAILURE;
        }

        status = graph->build_operation_graph(handle);
        if (!status.is_good()) {
            delete bn_graph;
            return FAILURE;
        }

        status = graph->create_execution_plans({HeurMode_t::FALLBACK});
        if (!status.is_good()) {
            delete bn_graph;
            return FAILURE;
        }

        status = graph->check_support(handle);
        if (!status.is_good()) {
            delete bn_graph;
            return FAILURE;
        }

        status = graph->build_plans(handle);
        if (!status.is_good()) {
            delete bn_graph;
            return FAILURE;
        }

        bn_graph->graph_ptr = graph;
        bn_graph->mode = mode;

        *graph_out = static_cast<BNGraph_t>(bn_graph);
        return SUCCESS;

    } catch (const std::exception& e) {
        return FAILURE;
    }
}

// Get workspace size
CudnnFrontendError_t get_bn_workspace_size(BNGraph_t graph, size_t* workspace_size) {
    if (graph == nullptr || workspace_size == nullptr) {
        return INVALID_VALUE;
    }
    BNGraph* bn_graph = static_cast<BNGraph*>(graph);
    try {
        int64_t ws_size;
        auto status = bn_graph->graph_ptr->get_workspace_size(ws_size);
        if (status.is_good()) {
            *workspace_size = static_cast<size_t>(ws_size);
            return SUCCESS;
        } else {
            return FAILURE;
        }
    } catch (const std::exception& e) {
        return FAILURE;
    }
}

// Execute BN graph
CudnnFrontendError_t execute_bn_graph(
    cudnnHandle_t handle,
    BNGraph_t graph,
    void* input_ptrs[],
    void* output_ptrs[],
    void* workspace) {
    if (graph == nullptr || input_ptrs == nullptr || output_ptrs == nullptr) {
        return INVALID_VALUE;
    }
    BNGraph* bn_graph = static_cast<BNGraph*>(graph);
    if (bn_graph->input_tensors.size() == 0 || bn_graph->output_tensors.size() == 0) {
        return INVALID_VALUE;
    }
    try {
        std::unordered_map<int64_t, void*> variant_pack;
        for (size_t i = 0; i < bn_graph->input_tensors.size(); ++i) {
            if (input_ptrs[i] == nullptr) {
                return INVALID_VALUE;
            }
            variant_pack[bn_graph->input_tensors[i]->get_uid()] = input_ptrs[i];
        }
        for (size_t i = 0; i < bn_graph->output_tensors.size(); ++i) {
            if (output_ptrs[i] == nullptr) {
                return INVALID_VALUE;
            }
            variant_pack[bn_graph->output_tensors[i]->get_uid()] = output_ptrs[i];
        }
        auto status = bn_graph->graph_ptr->execute(handle, variant_pack, workspace);
        if (status.is_good()) {
            return SUCCESS;
        } else {
            return FAILURE;
        }
    } catch (const std::exception& e) {
        return FAILURE;
    }
}

// Destroy BN graph
void destroy_bn_graph(BNGraph_t graph) {
    if (graph != nullptr) {
        BNGraph* bn_graph = static_cast<BNGraph*>(graph);
        delete bn_graph;
    }
}

// Get number of inputs
CudnnFrontendError_t get_bn_num_inputs(BNGraph_t graph, size_t* num_inputs) {
    if (graph == nullptr || num_inputs == nullptr) {
        return INVALID_VALUE;
    }
    BNGraph* bn_graph = static_cast<BNGraph*>(graph);
    *num_inputs = bn_graph->input_tensors.size();
    return SUCCESS;
}

// Get number of outputs
CudnnFrontendError_t get_bn_num_outputs(BNGraph_t graph, size_t* num_outputs) {
    if (graph == nullptr || num_outputs == nullptr) {
        return INVALID_VALUE;
    }
    BNGraph* bn_graph = static_cast<BNGraph*>(graph);
    *num_outputs = bn_graph->output_tensors.size();
    return SUCCESS;
}

} // extern "C"

