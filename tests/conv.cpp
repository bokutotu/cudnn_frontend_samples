#include <catch2/catch.hpp>

#include "cudnn_frontend_wrapper.h"
#include <cudnn.h>

#include <cstdlib>
#include "helpers.h"

TEST_CASE("conv2d", "[conv2d]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    CudnnTensorShapeStride shape = {
        .num_dims = 4,
        .dims = {4, 3, 32, 32},
        .strides = {3*32*32, 32*32, 32, 1}
    };
    CudnnTensorShapeStride filter_shape = {
        .num_dims = 4,
        .dims = {32, 3, 3, 3},
        .strides = {3*3*3, 3*3, 3, 1}
    };
    CudnnTensorShapeStride y_shape = {
        .num_dims = 4,
        .dims = {4, 32, 30, 30},
        .strides = {32*30*30, 30*30, 30, 1}
    };
    ConvInfo info = {
        .padding = {1, 1},
        .stride = {1, 1},
        .dilation = {1, 1},
        .num_dims = 2
    };
    ConvDescriptor* desc;
    CudnnFrontendError_t status = create_conv_descriptor(&desc, 
                                                         DATA_TYPE_FLOAT, 
                                                         &shape, 
                                                         &filter_shape, 
                                                         &y_shape, 
                                                         &info);
    REQUIRE(status == SUCCESS);
    CHECK(desc != nullptr);

    status = check_conv_graph(desc, &handle);
    REQUIRE(status == SUCCESS);

    int64_t workspace_size;
    status = get_conv_workspace_size(desc, &workspace_size);
    REQUIRE(status == SUCCESS);
    CHECK(workspace_size > 0);

    ConvBufers buffers = {
        .X = nullptr,
        .filter = nullptr,
        .Y = nullptr
    };

    Surface<float> X_tensor(4 * 3 * 32 * 32, false);
    Surface<float> Filter_tensor(32 * 3 * 3 * 3, false);
    Surface<float> Y_tensor(4 * 32 * 30 * 30, false);

    buffers.X = X_tensor.devPtr;
    buffers.filter = Filter_tensor.devPtr;
    buffers.Y = Y_tensor.devPtr;

    status = execute_conv_forward(desc, &buffers, nullptr, &handle);
    REQUIRE(status == SUCCESS);
};
