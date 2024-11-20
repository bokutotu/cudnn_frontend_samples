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

    Shape input_shape = {input_dim.data(), input_stride.data(), 3};
    Shape weight_shape = {weight_dim.data(), weight_stride.data(), 3};
    Shape output_shape = {output_dim.data(), output_stride.data(), 3};
    ConvParams conv_params = {padding.data(), stride.data(), dilation.data(), 1};

    auto forward_components = create_forward_graph(handle, &input_shape, &weight_shape, &output_shape, &conv_params);
    auto backward_filter_components = create_backward_filter_graph(handle, &input_shape, &weight_shape, &output_shape, &conv_params);
    auto backward_data_components = create_backward_data_graph(handle, &input_shape, &weight_shape, &output_shape, &conv_params);

    float* y_tensor = nullptr;
    cudaMalloc(&y_tensor, 4 * 64 * 16 * sizeof(float));
    float* dy_tensor = nullptr;
    cudaMalloc(&dy_tensor, 4 * 64 * 16 * sizeof(float));
    float* w_tensor = nullptr;
    cudaMalloc(&w_tensor, 64 * 32 * 3 * sizeof(float));
    float* dw_tensor = nullptr;
    cudaMalloc(&dw_tensor, 64 * 32 * 3 * sizeof(float));
    float* x_tensor = nullptr;
    cudaMalloc(&x_tensor, 4 * 32 * 16 * sizeof(float));
    float* dx_tensor = nullptr;
    cudaMalloc(&dx_tensor, 4 * 32 * 16 * sizeof(float));

    std::vector<float> y_init(4 * 64 * 16, 1.0f);
    std::vector<float> dy_init(4 * 64 * 16, 1.0f);
    std::vector<float> w_init(64 * 32 * 3, 2.0f);
    std::vector<float> dw_init(64 * 32 * 3, 0.0f);
    std::vector<float> x_init(4 * 32 * 16, 0.0f);
    std::vector<float> dx_init(4 * 32 * 16, 0.0f);

    cudaMemcpy(y_tensor, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy_tensor, dy_init.data(), dy_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_tensor, w_init.data(), w_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw_tensor, dw_init.data(), dw_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_tensor, x_init.data(), x_init.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx_tensor, dx_init.data(), dx_init.size() * sizeof(float), cudaMemcpyHostToDevice);

    copy_and_print_tensor(y_tensor, 4 * 64 * 16, "Y");
    copy_and_print_tensor(dy_tensor, 4 * 64 * 16, "DY");
    copy_and_print_tensor(w_tensor, 64 * 32 * 3, "W");
    copy_and_print_tensor(dw_tensor, 64 * 32 * 3, "DW");
    copy_and_print_tensor(x_tensor, 4 * 32 * 16, "X");
    copy_and_print_tensor(dx_tensor, 4 * 32 * 16, "DX");

    // Execute Forward
    if (execute_graph(handle, forward_components, dy_tensor, w_tensor, y_tensor)) {
        copy_and_print_tensor(y_tensor, 4 * 32 * 16, "Y (after forward)");
    }

    // Execute Backward Filter
    if (execute_graph(handle, backward_filter_components, dy_tensor, x_tensor, dw_tensor)) {
        copy_and_print_tensor(dw_tensor, 64 * 32 * 3, "DW (after backward filter)");
    }

    // Execute Backward Data
    if (execute_graph(handle, backward_data_components, dy_tensor, w_tensor, dx_tensor)) {
        copy_and_print_tensor(dx_tensor, 4 * 32 * 16, "DX (after backward data)");
    }

    // print results
    copy_and_print_tensor(y_tensor, 4 * 64 * 16, "Y");
    copy_and_print_tensor(dx_tensor, 4 * 32 * 16, "DX");
    copy_and_print_tensor(dw_tensor, 64 * 32 * 3, "DW");


    cudnnDestroy(handle);
    destroy_graph_components(forward_components);
    destroy_graph_components(backward_filter_components);
    destroy_graph_components(backward_data_components);
    cudaFree(y_tensor);
    cudaFree(dy_tensor);
    cudaFree(w_tensor);
    cudaFree(dw_tensor);
    cudaFree(x_tensor);
    cudaFree(dx_tensor);
    return 0;
}
