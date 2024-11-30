#include <iostream>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "cudnn_frontend.h"

#define REQUIRE(x)                                                  \
    {                                                               \
        if (!(x)) {                                                 \
            std::cerr << "Error at line " << __LINE__ << std::endl; \
            return 1;                                               \
        }                                                           \
    }


#define CUDA_CHECK(status)                                                                                    \
    {                                                                                                         \
        cudaError_t err = status;                                                                             \
        if (err != cudaSuccess) {                                                                             \
            std::stringstream err_msg;                                                                        \
            err_msg << "CUDA Error: " << cudaGetErrorString(err) << " (" << err << ") at " << __FILE__ << ":" \
                    << __LINE__;                                                                              \
            std::cerr << err_msg.str() << std::endl;                                                          \
            throw std::runtime_error(err_msg.str());                                                          \
        }                                                                                                     \
    }

#define CUDNN_CHECK(status)                                                                                     \
    {                                                                                                           \
        cudnnStatus_t err = status;                                                                             \
        if (err != CUDNN_STATUS_SUCCESS) {                                                                      \
            std::stringstream err_msg;                                                                          \
            err_msg << "cuDNN Error: " << cudnnGetErrorString(err) << " (" << err << ") at " << __FILE__ << ":" \
                    << __LINE__;                                                                                \
            std::cerr << err_msg.str() << std::endl;                                                            \
            throw std::runtime_error(err_msg.str());                                                            \
        }                                                                                                       \
    }


template <typename T_ELEM>
T_ELEM* alloc_gpu(int64_t n_elems) {
    T_ELEM* devPtr;
    CUDA_CHECK(
        cudaMalloc((void**)&(devPtr), (size_t)((n_elems) * sizeof(devPtr[0])))
    );
    return devPtr;
}

int main() {
    // CUDA_CHECK(cudaSetDevice(0));
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    bool has_running_stats = true;
    auto X                 = graph.tensor(fe::graph::Tensor_attributes()
                              // .set_name("X")
                              .set_dim({4, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));
    auto prev_running_mean = graph.tensor(fe::graph::Tensor_attributes()
                                              // .set_name("prev_running_mean")
                                              .set_dim({1, 32, 1, 1})
                                              .set_stride({32, 1, 32, 32})
                                              .set_data_type(fe::DataType_t::FLOAT));
    auto prev_running_var  = graph.tensor(fe::graph::Tensor_attributes()
                                             // .set_name("prev_running_var")
                                             .set_dim({1, 32, 1, 1})
                                             .set_stride({32, 1, 32, 32})
                                             .set_data_type(fe::DataType_t::FLOAT));
    auto scale             = graph.tensor(fe::graph::Tensor_attributes()
                                  // .set_name("scale")
                                  .set_dim({1, 32, 1, 1})
                                  .set_stride({32, 1, 32, 32})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias              = graph.tensor(fe::graph::Tensor_attributes()
                                 // .set_name("bias")
                                 .set_dim({1, 32, 1, 1})
                                 .set_stride({32, 1, 32, 32})
                                 .set_data_type(fe::DataType_t::FLOAT));

    auto peer_stats_0 = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_dim({2, 4 * 32, 1, 1})
                                         .set_stride({4 * 32, 1, 4 * 32, 4 * 32})
                                         .set_data_type(fe::DataType_t::FLOAT));
    auto peer_stats_1 = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_dim({2, 4 * 32, 1, 1})
                                         .set_stride({4 * 32, 1, 4 * 32, 4 * 32})
                                         .set_data_type(fe::DataType_t::FLOAT));

    auto epsilon  = graph.tensor(1e-05f);
    auto momentum = graph.tensor(1e-01f);

    auto batchnorm_options =
        fe::graph::Batchnorm_attributes().set_epsilon(epsilon).set_peer_stats({peer_stats_0, peer_stats_1});
    if (has_running_stats) {
        batchnorm_options.set_previous_running_stats(prev_running_mean, prev_running_var, momentum);
    }

    auto [bn_output, mean, inv_variance, next_running_mean, next_running_var] =
        graph.batchnorm(X, scale, bias, batchnorm_options);
    mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    if (has_running_stats) {
        next_running_mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        next_running_var->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    }

    auto A           = graph.tensor(fe::graph::Tensor_attributes()
                              // .set_name("A")
                              .set_dim({4, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                              .set_data_type(fe::DataType_t::HALF));
    auto add_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto add_output  = graph.pointwise(bn_output, A, add_options);

    auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto Y            = graph.pointwise(add_output, relu_options);
    Y->set_output(true);

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());


    std::cout << "set up graph complited" << std::endl;

    auto X_tensor = alloc_gpu<half>(4 * 32 * 16 * 16);
    auto Mean_tensor = alloc_gpu<float>(32);
    auto Var_tensor = alloc_gpu<float>(32);
    auto Previous_running_mean_tensor = alloc_gpu<float>(32);
    auto Previous_running_var_tensor = alloc_gpu<float>(32);
    auto Next_running_mean_tensor = alloc_gpu<float>(32);
    auto Next_running_var_tensor = alloc_gpu<float>(32);
    auto Scale_tensor = alloc_gpu<float>(32);
    auto Bias_tensor = alloc_gpu<float>(32);
    auto A_tensor = alloc_gpu<half>(4 * 32 * 16 * 16);
    auto Y_tensor = alloc_gpu<half>(4 * 32 * 16 * 16);
    auto Peer_stats_0_tensor = alloc_gpu<float>(2 * 4 * 32);
    auto Peer_stats_1_tensor = alloc_gpu<float>(2 * 4 * 32);


    int64_t workspace_size;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    auto workspace = alloc_gpu<int8_t>(workspace_size);

    std::cout << "allocated memory" << std::endl;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor},
        {mean, Mean_tensor},
        {inv_variance, Var_tensor},
        {scale, Scale_tensor},
        {bias, Bias_tensor},
        {A, A_tensor},
        {Y, Y_tensor},
        {peer_stats_0, Peer_stats_0_tensor},
        {peer_stats_1, Peer_stats_1_tensor}};

    if (has_running_stats) {
        variant_pack[prev_running_mean] = Previous_running_mean_tensor;
        variant_pack[prev_running_var]  = Previous_running_var_tensor;
        variant_pack[next_running_mean] = Next_running_mean_tensor;
        variant_pack[next_running_var]  = Next_running_var_tensor;
    }
    std::cout << "before execute" << std::endl;
    REQUIRE(graph.execute(handle, variant_pack, workspace).is_good());
    std::cout << "after execute" << std::endl;

    CUDA_CHECK(cudaFree(X_tensor));
    CUDA_CHECK(cudaFree(Mean_tensor));
    CUDA_CHECK(cudaFree(Var_tensor));
    CUDA_CHECK(cudaFree(Previous_running_mean_tensor));
    CUDA_CHECK(cudaFree(Previous_running_var_tensor));
    CUDA_CHECK(cudaFree(Next_running_mean_tensor));
    CUDA_CHECK(cudaFree(Next_running_var_tensor));
    CUDA_CHECK(cudaFree(Scale_tensor));
    CUDA_CHECK(cudaFree(Bias_tensor));
    CUDA_CHECK(cudaFree(A_tensor));
    CUDA_CHECK(cudaFree(Y_tensor));
    CUDA_CHECK(cudaFree(Peer_stats_0_tensor));
    CUDA_CHECK(cudaFree(Peer_stats_1_tensor));
    CUDA_CHECK(cudaFree(workspace));

    std::cout << "freed memory" << std::endl;

    cudaDeviceSynchronize();
    std::cout << "sync done" << std::endl;

    cudnnDestroy(handle);

    return 0;
}
