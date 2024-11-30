#include <iostream>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "cudnn_frontend.h"

struct BatchNormTensorDesc {
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> X;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> prev_running_mean;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> prev_running_var;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> scale;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> bias;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> peer_stats_0;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> peer_stats_1;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> mean;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> inv_variance;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> next_running_mean;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> next_running_var;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> epsilon;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> momentum;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Y;
};

struct BatchNormBuilder {
    cudnn_frontend::DataType_t io_data_type;
    cudnn_frontend::DataType_t intermediate_data_type;
    cudnn_frontend::DataType_t compute_data_type;

    int input_dim_len;
    int64_t input_dim[8];

    float epsilon;
    float momentum;
};

std::vector<int64_t> get_default_stride(int len, int64_t dim[8]) {
    std::vector<int64_t> stride(len);
    stride[len - 1] = 1;
    for (int i = len - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * dim[i + 1];
    }
    return stride;
}

std::vector<int64_t> get_static_shape(int len, int64_t dim[8]) {
    if (len == 2) {
        return {1, dim[1]};
    } else if (len == 4) {
        return {1, dim[1], 1, 1};
    } else {
        throw std::runtime_error("Unsupported dimension length");
    }
}

std::vector<int64_t> get_input_shape_vector(int len, int64_t dim[8]) {
    std::vector<int64_t> shape(len);
    for (int i = 0; i < len; i++) {
        shape[i] = dim[i];
    }
    return shape;
}

std::pair<BatchNormTensorDesc, cudnn_frontend::graph::Graph> create_batchnorm_graph(BatchNormBuilder builder) {
    cudnn_frontend::graph::Graph graph;
    graph.set_io_data_type(builder.io_data_type)
         .set_intermediate_data_type(builder.intermediate_data_type)
         .set_compute_data_type(builder.compute_data_type);

    auto input_shape = get_input_shape_vector(builder.input_dim_len, builder.input_dim);
    auto input_stride = get_default_stride(input_shape.size(), input_shape.data());

    auto static_shape = get_static_shape(builder.input_dim_len, builder.input_dim);
    auto static_stride = get_default_stride(static_shape.size(), static_shape.data());

    auto X = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                            .set_name("X")
                            .set_dim(input_shape)
                            .set_stride(input_stride)
                            .set_data_type(builder.io_data_type));

    auto prev_running_mean = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                                            .set_name("prev_running_mean")
                                            .set_dim(static_shape)
                                            .set_stride(static_stride)
                                            .set_data_type(builder.io_data_type));

    auto prev_running_var = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                                            .set_name("prev_running_var")
                                            .set_dim(static_shape)
                                            .set_stride(static_stride)
                                            .set_data_type(builder.io_data_type));

    auto scale = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                                .set_name("scale")
                                .set_dim(static_shape)
                                .set_stride(static_stride)
                                .set_data_type(builder.io_data_type));

    auto bias = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                                .set_name("bias")
                                .set_dim(static_shape)
                                .set_stride(static_stride)
                                .set_data_type(builder.io_data_type));

    auto peer_stats_0 = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                                .set_dim(static_shape)
                                .set_stride(static_stride)
                                .set_data_type(builder.io_data_type));

    auto peer_stats_1 = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                                        .set_dim(static_shape)
                                        .set_stride(static_stride)
                                        .set_data_type(builder.io_data_type));

    auto epsilon = graph.tensor(builder.epsilon);
    auto momentum = graph.tensor(builder.momentum);

    auto batchnorm_options = cudnn_frontend::graph::Batchnorm_attributes()
                                .set_epsilon(epsilon)
                                .set_previous_running_stats(prev_running_mean, prev_running_var, momentum);

    auto [Y, mean, inv_variance, next_running_mean, next_running_var] =
        graph.batchnorm(X, scale, bias, batchnorm_options);
    mean->set_output(true).set_data_type(builder.io_data_type);
    inv_variance->set_output(true).set_data_type(builder.io_data_type);

    next_running_mean->set_output(true).set_data_type(builder.io_data_type);
    next_running_var->set_output(true).set_data_type(builder.io_data_type);

    BatchNormTensorDesc desc = {
        X, prev_running_mean, prev_running_var, scale, bias, peer_stats_0, peer_stats_1,
        mean, inv_variance, next_running_mean, next_running_var, epsilon, momentum, Y
    };
    return {desc, graph};
}

void check_graph(cudnnHandle_t handle, cudnn_frontend::graph::Graph graph) {
    auto err = graph.validate();
    if (!err.is_good()) {
        throw std::runtime_error("Error validating graph");
    }

    err = graph.build_operation_graph(handle);
    if (!err.is_good()) {
        throw std::runtime_error("Error building operation graph");
    }

    err = graph.create_execution_plans({cudnn_frontend::HeurMode_t::FALLBACK});
    if (!err.is_good()) {
        throw std::runtime_error("Error creating execution plans");
    }

    err = graph.check_support(handle);
    if (!err.is_good()) {
        throw std::runtime_error("Error checking support");
    }

    err = graph.build_plans(handle);
    if (!err.is_good()) {
        throw std::runtime_error("Error building plans");
    }
}

int64_t get_batch_norm_workspace_size(cudnn_frontend::graph::Graph graph) {
    int64_t workspace_size;
    auto err = graph.get_workspace_size(workspace_size);
    if (!err.is_good()) {
        throw std::runtime_error("Error getting workspace size");
    }
    return workspace_size;
}

struct BatchNormExecutePointers {
    float *X;
    float *prev_running_mean;
    float *prev_running_var;
    float *scale;
    float *bias;
    float *peer_stats_0;
    float *peer_stats_1;
    float *mean;
    float *inv_variance;
    float *next_running_mean;
    float *next_running_var;
    float *epsilon;
    float *momentum;
    float *Y;
    float *workspace;
};

void execute_batchnorm_graph(cudnnHandle_t handle, BatchNormTensorDesc desc, cudnn_frontend::graph::Graph graph, BatchNormExecutePointers pointers) {
    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void *> variant_pack = {
        {desc.X, pointers.X},
        {desc.prev_running_mean, pointers.prev_running_mean},
        {desc.prev_running_var, pointers.prev_running_var},
        {desc.scale, pointers.scale},
        {desc.bias, pointers.bias},
        {desc.peer_stats_0, pointers.peer_stats_0},
        {desc.peer_stats_1, pointers.peer_stats_1},
        {desc.mean, pointers.mean},
        {desc.inv_variance, pointers.inv_variance},
        {desc.next_running_mean, pointers.next_running_mean},
        {desc.next_running_var, pointers.next_running_var},
        {desc.epsilon, pointers.epsilon},
        {desc.momentum, pointers.momentum},
        {desc.Y, pointers.Y}
    };

    auto err = graph.execute(handle, variant_pack, pointers.workspace);
}

int main() {
    return 0;
}
