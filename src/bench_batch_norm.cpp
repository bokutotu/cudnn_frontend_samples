#include "cudnn_frontend.h"
#include <iostream>
#include <chrono>

struct BNTensors {
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> sum;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> sq_sum;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> scale;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> bias;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> prev_running_mean;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> prev_running_var;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> eq_scale;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> eq_bias;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> saved_mean;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> saved_inv_variance;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> next_running_mean;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> next_running_var;
};

std::tuple<cudnn_frontend::graph::Graph, BNTensors> create_bn_graph() {
    cudnn_frontend::graph::Graph graph;
    BNTensors tensors;
    
    graph.set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
        .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
        .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    tensors.sum = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                               .set_name("sum")
                               .set_dim({1, 32, 1, 1})
                               .set_stride({32, 1, 32, 32}));
    tensors.sq_sum = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                                  .set_name("sq_sum")
                                  .set_dim({1, 32, 1, 1})
                                  .set_stride({32, 1, 32, 32}));
    tensors.prev_running_mean = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                                             .set_name("prev_running_mean")
                                             .set_dim({1, 32, 1, 1})
                                             .set_stride({32, 1, 32, 32}));
    tensors.prev_running_var = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                                            .set_name("prev_running_var")
                                            .set_dim({1, 32, 1, 1})
                                            .set_stride({32, 1, 32, 32}));
    tensors.scale = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                                 .set_name("scale")
                                 .set_dim({1, 32, 1, 1})
                                 .set_stride({32, 1, 32, 32}));
    tensors.bias = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                                .set_name("bias")
                                .set_dim({1, 32, 1, 1})
                                .set_stride({32, 1, 32, 32}));

    float EPS_scalar = 0.001f;
    float MOMENTUM_scalar = 0.001f;
    int64_t nhw = 64;

    auto epsilon = graph.tensor(EPS_scalar);
    auto momentum = graph.tensor(MOMENTUM_scalar);
    auto accum_count = graph.tensor(nhw);

    auto bn_finalize_options = cudnn_frontend::graph::BN_finalize_attributes()
        .set_previous_running_stats(tensors.prev_running_mean, tensors.prev_running_var, momentum);
    
    auto [eq_scale, eq_bias, saved_mean, saved_inv_variance, next_running_mean, next_running_var] =
        graph.bn_finalize(tensors.sum, tensors.sq_sum, tensors.scale, tensors.bias, 
                         epsilon, accum_count, bn_finalize_options);

    tensors.eq_scale = eq_scale;
    tensors.eq_bias = eq_bias;
    tensors.saved_mean = saved_mean;
    tensors.saved_inv_variance = saved_inv_variance;
    tensors.next_running_mean = next_running_mean;
    tensors.next_running_var = next_running_var;

    eq_scale->set_output(true);
    eq_bias->set_output(true);
    saved_mean->set_output(true);
    saved_inv_variance->set_output(true);
    next_running_mean->set_output(true);
    next_running_var->set_output(true);

    return {graph, tensors};
}

void execute_bn_graph(cudnn_frontend::graph::Graph& graph, const BNTensors& tensors) {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    void *d_sum, *d_sq_sum, *d_mean, *d_var, *d_prev_mean, *d_prev_var;
    void *d_next_mean, *d_next_var, *d_scale, *d_bias, *d_eq_scale, *d_eq_bias;
    
    const size_t tensor_size = 32 * sizeof(float);
    cudaMalloc((void**)&d_sum, tensor_size);
    cudaMalloc((void**)&d_sq_sum, tensor_size);
    cudaMalloc((void**)&d_mean, tensor_size);
    cudaMalloc((void**)&d_var, tensor_size);
    cudaMalloc((void**)&d_prev_mean, tensor_size);
    cudaMalloc((void**)&d_prev_var, tensor_size);
    cudaMalloc((void**)&d_next_mean, tensor_size);
    cudaMalloc((void**)&d_next_var, tensor_size);
    cudaMalloc((void**)&d_scale, tensor_size);
    cudaMalloc((void**)&d_bias, tensor_size);
    cudaMalloc((void**)&d_eq_scale, tensor_size);
    cudaMalloc((void**)&d_eq_bias, tensor_size);

    auto err = graph.validate();
    if (!err.is_good()) {
        std::cerr << "Error: " << err << std::endl;
        return;
    }
    err = graph.build_operation_graph(handle);
    if (!err.is_good()) {
        std::cerr << "Error: " << err << std::endl;
        return;
    }
    err = graph.create_execution_plans({cudnn_frontend::HeurMode_t::FALLBACK});
    if (!err.is_good()) {
        std::cerr << "Error: " << err << std::endl;
        return;
    }
    err = graph.check_support(handle);
    if (!err.is_good()) {
        std::cerr << "Error: " << err << std::endl;
        return;
    }
    err = graph.build_plans(handle);
    if (!err.is_good()) {
        std::cerr << "Error: " << err << std::endl;
        return;
    }

    int64_t workspace_size;
    err = graph.get_workspace_size(workspace_size);
    void* d_workspace;
    cudaMalloc((void**)&d_workspace, workspace_size);

    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> variant_pack = {
        {tensors.sum, d_sum},
        {tensors.sq_sum, d_sq_sum},
        {tensors.scale, d_scale},
        {tensors.bias, d_bias},
        {tensors.prev_running_mean, d_prev_mean},
        {tensors.prev_running_var, d_prev_var},
        {tensors.eq_scale, d_eq_scale},
        {tensors.eq_bias, d_eq_bias},
        {tensors.saved_mean, d_mean},
        {tensors.saved_inv_variance, d_var},
        {tensors.next_running_mean, d_next_mean},
        {tensors.next_running_var, d_next_var}
    };

    err = graph.execute(handle, variant_pack, d_workspace);
    if (!err.is_good()) {
        std::cerr << "Error: " << err << std::endl;
        return;
    }

    cudaFree(d_sum);
    cudaFree(d_sq_sum);
    cudaFree(d_mean);
    cudaFree(d_var);
    cudaFree(d_prev_mean);
    cudaFree(d_prev_var);
    cudaFree(d_next_mean);
    cudaFree(d_next_var);
    cudaFree(d_scale);
    cudaFree(d_bias);
    cudaFree(d_eq_scale);
    cudaFree(d_eq_bias);
    cudaFree(d_workspace);
    
    cudnnDestroy(handle);
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 2; i++) {
        auto [graph, tensors] = create_bn_graph();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    auto [graph, tensors] = create_bn_graph();
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 2; i++) {
        execute_bn_graph(graph, tensors);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    return 0;
}
