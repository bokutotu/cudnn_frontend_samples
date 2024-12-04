#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "cudnn_frontend_wrapper.h"
#include <cudnn.h>

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
    CudnnFrontendError_t status = create_batch_norm_descriptor(&desc, DATA_TYPE_FLOAT, &shape, epsilon, momentum, is_training);
    // batch_norm_desc_debug(desc);
    REQUIRE(status == SUCCESS);
    CHECK(desc != nullptr);
    status = check_graph(desc, &handle);
    REQUIRE(status == SUCCESS);
    int64_t workspace_size;
    status = get_workspace_size(desc, &workspace_size);
    REQUIRE(status == SUCCESS);
    CHECK(workspace_size > 0);
    std::cout << "Workspace size: " << workspace_size << std::endl;
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
    cudaMalloc(&buffers.X, 4*32*16*16*sizeof(float));
    cudaMalloc(&buffers.mean, 32*sizeof(float));
    cudaMalloc(&buffers.inv_variance, 32*sizeof(float));
    cudaMalloc(&buffers.scale, 32*sizeof(float));
    cudaMalloc(&buffers.bias, 32*sizeof(float));
    cudaMalloc(&buffers.peer_stats_0, 2*4*32*sizeof(float));
    cudaMalloc(&buffers.peer_stats_1, 2*4*32*sizeof(float));
    cudaMalloc(&buffers.prev_running_mean, 32*sizeof(float));
    cudaMalloc(&buffers.prev_running_var, 32*sizeof(float));
    cudaMalloc(&buffers.next_running_mean, 32*sizeof(float));
    cudaMalloc(&buffers.next_running_var, 32*sizeof(float));
    cudaMalloc(&buffers.Y, 4*32*16*16*sizeof(float));
    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);
    status = execute_batch_norm_forward_training(desc, &buffers, workspace, &handle);
    REQUIRE(status == SUCCESS);
    std::cout << "BatchNorm forward training executed successfully" << std::endl;
};
