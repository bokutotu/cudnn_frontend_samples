#include <iostream>
#include <vector>
#include <memory>
#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <unordered_map>

#include "conv.h"

#define CUDNN_CHECK(status)                                                                                     \
    {                                                                                                           \
        cudnnStatus_t err = status;                                                                             \
        if (err != CUDNN_STATUS_SUCCESS) {                                                                      \
            std::stringstream err_msg;                                                                          \
            err_msg << "cuDNN Error: " << cudnnGetErrorString(err) << " (" << err << ") at " << __FILE__ << ":" \
                    << __LINE__;                                                                                \
            return 1;                                                                                           \
        }                                                                                                       \
    }

void print_tensor(const float* tensor, int size, const std::string& name) {
    std::cout << name << ": ";
    for (int i = 0; i < size; ++i) {
        std::cout << tensor[i] << " ";
    }
    std::cout << std::endl;
}

void copy_and_print_tensor(float* device_tensor, int size, const std::string& name) {
    std::vector<float> host_tensor(size);
    cudaMemcpy(host_tensor.data(), device_tensor, size * sizeof(float), cudaMemcpyDeviceToHost);
    print_tensor(host_tensor.data(), size, name);
}

int main() {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    std::vector<long int> input_dim = {4, 64, 16};
    std::vector<long int> weight_dim = {64, 32, 3};
    std::vector<long int> output_dim = {4, 32, 16};
    std::vector<long int> input_stride = {64 * 16, 1, 64};
    std::vector<long int> weight_stride = {32 * 3, 1, 32};
    std::vector<long int> output_stride = {32 * 16, 1, 32};
    std::vector<long int> padding = {1};
    std::vector<long int> stride = {1};
    std::vector<long int> dilation = {1};
    auto input_shape = Shape{.dim = input_dim.data(), .stride = input_stride.data(), .size = 3};
    auto weight_shape = Shape{.dim = weight_dim.data(), .stride = weight_stride.data(), .size = 3};
    auto output_shape = Shape{.dim = output_dim.data(), .stride = output_stride.data(), .size = 3};
    auto conv_params = ConvParams{.padding = padding.data(), .stride = stride.data(), .dilation = dilation.data(), .size = 1};
    auto components = create_graph(&input_shape, &weight_shape, &output_shape, &conv_params);

    float* dy_tensor = nullptr;
    cudaMalloc(&dy_tensor, 4 * 64 * 16 * sizeof(float));
    float* w_tensor = nullptr;
    cudaMalloc(&w_tensor, 64 * 32 * 3 * sizeof(float));
    float* dx_tensor = nullptr;
    cudaMalloc(&dx_tensor, 4 * 32 * 16 * sizeof(float));

    std::vector<float> dy_init(4 * 64 * 16, 1.0f);
    std::vector<float> w_init(64 * 32 * 3, 2.0f);
    std::vector<float> dx_init(4 * 32 * 16, 0.0f);
    cudaMemcpy(dy_tensor, dy_init.data(), dy_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_tensor, w_init.data(), w_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx_tensor, dx_init.data(), dx_init.size() * sizeof(float), cudaMemcpyHostToDevice);

    copy_and_print_tensor(dy_tensor, 4 * 64 * 16, "DY");
    copy_and_print_tensor(w_tensor, 64 * 32 * 3, "W");
    copy_and_print_tensor(dx_tensor, 4 * 32 * 16, "DX");

    if (execute_graph(handle, components, dy_tensor, w_tensor, dx_tensor)) {
        copy_and_print_tensor(dx_tensor, 4 * 32 * 16, "DX (after computation)");
    }

    cudnnDestroy(handle);
    destroy_graph_components(components);
    return 0;
}
