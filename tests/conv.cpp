#include <catch2/catch.hpp>

#include "cudnn_frontend_wrapper.h"
#include <cudnn.h>

#include <cstdlib>
#include "helpers.h"

TEST_CASE("conv2d", "[conv2d]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

    CudnnTensorShapeStride shape = {
        .num_dims = 4,
        .dims = {n, c, h, w},
        .strides = {c * h * w, h * w, w, 1}
    };
    CudnnTensorShapeStride filter_shape = {
        .num_dims = 4,
        .dims = {k, c, r, s},
        .strides = {c * r * s, r * s, s, 1}
    };
    CudnnTensorShapeStride y_shape = {
        .num_dims = 4,
        .dims = {n, k, h, w},
        .strides = {k * h * w, h * w, w, 1}
    };
    ConvInfo info = {
        .padding = {0, 0},
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

    ConvBufers buffers = {
        .X = nullptr,
        .filter = nullptr,
        .Y = nullptr
    };

    Surface<float> X_tensor(n * c * h * w, false);
    Surface<float> Filter_tensor(k * c * r * s, false);
    Surface<float> Y_tensor(n * k * h * w, false);
    Surface<int8_t> workspace(workspace_size, false);

    buffers.X = X_tensor.devPtr;
    buffers.filter = Filter_tensor.devPtr;
    buffers.Y = Y_tensor.devPtr;

    status = execute_conv_forward(desc, &buffers, workspace.devPtr, &handle);
    REQUIRE(status == SUCCESS);
};
