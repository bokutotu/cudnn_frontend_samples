#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "cudnn_frontend_wrapper.h"
#include <cudnn.h>

#include <cstdlib>
#include "helpers.h"

void init_tensor_in_cpu(float* tensor, int64_t size) {
    for (int64_t i = 0; i < size; i++) {
        tensor[i] = (float)rand() / 1000.;
    }
}

void init_tensor_in_gpu(float* tensor, int64_t size) {
    float* tensor_cpu = (float*)malloc(size * sizeof(float));
    init_tensor_in_cpu(tensor_cpu, size);
    cudaMemcpy(tensor, tensor_cpu, size * sizeof(float), cudaMemcpyHostToDevice);
    free(tensor_cpu);
}

TEST_CASE("Example test case", "[batchnorm2d]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    CudnnTensorShapeStride shape = {
        .num_dims = 4,
        .dims = {4, 32, 16, 16},
        .strides = {32*16*16, 16*16, 16, 1}
    };
    float epsilon = 1e-5;
    float momentum = 0.1;
    bool is_training = true;
    BatchNormDescriptor* desc;
    CudnnFrontendError_t status = create_batch_norm_descriptor(&desc, 
                                                               DATA_TYPE_FLOAT, 
                                                               &shape, 
                                                               epsilon, 
                                                               momentum, 
                                                               is_training);
    REQUIRE(status == SUCCESS);
    CHECK(desc != nullptr);
    status = check_graph(desc, &handle);
    REQUIRE(status == SUCCESS);
    int64_t workspace_size;
    status = get_workspace_size(desc, &workspace_size);
    REQUIRE(status == SUCCESS);
    CHECK(workspace_size > 0);

    BatchNormExecutionBuffers buffers = {
        .X = nullptr,
        .mean = nullptr,
        .inv_variance = nullptr,
        .scale = nullptr,
        .bias = nullptr,
        .peer_stats_0 = nullptr,
        .peer_stats_1 = nullptr,
        .prev_running_mean = nullptr,
        .prev_running_var = nullptr,
        .next_running_mean = nullptr,
        .next_running_var = nullptr,
        .Y = nullptr
    };

    // cudaMalloc(&buffers.X, 4*32*16*16*sizeof(float));
    // cudaMalloc(&buffers.mean, 32*sizeof(float));
    // cudaMalloc(&buffers.inv_variance, 32*sizeof(float));
    // cudaMalloc(&buffers.scale, 32*sizeof(float));
    // cudaMalloc(&buffers.bias, 32*sizeof(float));
    // cudaMalloc(&buffers.peer_stats_0, 2*4*32*sizeof(float));
    // cudaMalloc(&buffers.peer_stats_1, 2*4*32*sizeof(float));
    // cudaMalloc(&buffers.prev_running_mean, 32*sizeof(float));
    // cudaMalloc(&buffers.prev_running_var, 32*sizeof(float));
    // cudaMalloc(&buffers.next_running_mean, 32*sizeof(float));
    // cudaMalloc(&buffers.next_running_var, 32*sizeof(float));
    // cudaMalloc(&buffers.Y, 4*32*16*16*sizeof(float));
    //
    // init_tensor_in_gpu((float*)buffers.X, 4*32*16*16);
    // init_tensor_in_gpu((float*)buffers.mean, 32);
    // init_tensor_in_gpu((float*)buffers.inv_variance, 32);
    // init_tensor_in_gpu((float*)buffers.scale, 32);
    // init_tensor_in_gpu((float*)buffers.bias, 32);
    // init_tensor_in_gpu((float*)buffers.peer_stats_0, 2*4*32);
    // init_tensor_in_gpu((float*)buffers.peer_stats_1, 2*4*32);
    // init_tensor_in_gpu((float*)buffers.prev_running_mean, 32);
    // init_tensor_in_gpu((float*)buffers.prev_running_var, 32);
    // init_tensor_in_gpu((float*)buffers.next_running_mean, 32);
    // init_tensor_in_gpu((float*)buffers.next_running_var, 32);
    // init_tensor_in_gpu((float*)buffers.Y, 4*32*16*16);
    Surface<float> X_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Var_tensor(32, false);
    Surface<float> Previous_running_mean_tensor(32, false);
    Surface<float> Previous_running_var_tensor(32, false);
    Surface<float> Next_running_mean_tensor(32, false);
    Surface<float> Next_running_var_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Bias_tensor(32, false);
    Surface<float> A_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Y_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Peer_stats_0_tensor(2 * 4 * 32, false, true);
    Surface<float> Peer_stats_1_tensor(2 * 4 * 32, false);
    buffers.X = X_tensor.devPtr;
    buffers.mean = Mean_tensor.devPtr;
    buffers.inv_variance = Var_tensor.devPtr;
    buffers.scale = Scale_tensor.devPtr;
    buffers.bias = Bias_tensor.devPtr;
    buffers.peer_stats_0 = Peer_stats_0_tensor.devPtr;
    buffers.peer_stats_1 = Peer_stats_1_tensor.devPtr;
    buffers.prev_running_mean = Previous_running_mean_tensor.devPtr;
    buffers.prev_running_var = Previous_running_var_tensor.devPtr;
    buffers.next_running_mean = Next_running_mean_tensor.devPtr;
    buffers.next_running_var = Next_running_var_tensor.devPtr;
    buffers.Y = Y_tensor.devPtr;

    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);
    cudaDeviceSynchronize();
    status = execute_batch_norm_forward_training(desc, &buffers, workspace, &handle);
    REQUIRE(status == SUCCESS);
};
