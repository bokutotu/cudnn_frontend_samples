#include "conv.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#include <string>

// Include the print_tensor and copy_and_print_tensor functions
void print_tensor(const float* tensor, int size, const std::string& name);
void copy_and_print_tensor(float* device_tensor, int size, const std::string& name);

int main() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    ConvError_t status;

    // Set tensor descriptors for the forward pass
    ConvTensorDescriptor_t input_desc = {
        .num_dims = 4,
        .dims = {1, 3, 5, 5}, // Example dimensions: N=1, C=3, H=5, W=5
        .strides = {3 * 5 * 5, 1, 5 * 5, 5}
    };

    ConvTensorDescriptor_t filter_desc = {
        .num_dims = 4,
        .dims = {2, 3, 3, 3}, // K=2 output channels
        .strides = {3 * 3 * 3, 1, 3 * 3, 3}
    };

    ConvTensorDescriptor_t output_desc = {
        .num_dims = 4,
        .dims = {1, 2, 3, 3},
        .strides = {2 * 3 * 3, 1, 3 * 3, 3}
    };

    ConvConvolutionDescriptor_t conv_desc = {
        .num_dims = 2,
        .padding = {0, 0},
        .stride = {1, 1},
        .dilation = {1, 1}
    };

    // Build forward propagation graph
    ConvGraph_t fwd_graph;
    status = build_fprop_graph(handle, &fwd_graph, &input_desc, &filter_desc, &output_desc, &conv_desc, CONV_DATA_TYPE_FLOAT);
    if (status != CONV_SUCCESS) {
        std::cerr << "Failed to build forward propagation graph." << std::endl;
        return -1;
    }

    // Allocate memory for tensors
    size_t input_size = 1 * 3 * 5 * 5;
    size_t filter_size = 2 * 3 * 3 * 3;
    size_t output_size = 1 * 2 * 3 * 3;

    float* d_input;
    float* d_filter;
    float* d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_filter, filter_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    // Initialize input and filter tensors on host
    std::vector<float> h_input(input_size, 1.0f);   // Initialize all elements to 1.0f
    std::vector<float> h_filter(filter_size, 1.0f); // Initialize all elements to 1.0f

    // Copy data to device
    cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter.data(), filter_size * sizeof(float), cudaMemcpyHostToDevice);

    // Get workspace size
    size_t workspace_size;
    status = get_workspace_size(fwd_graph, &workspace_size);
    if (status != CONV_SUCCESS) {
        std::cerr << "Failed to get workspace size for forward graph." << std::endl;
        destroy_graph(fwd_graph);
        return -1;
    }

    void* workspace;
    cudaMalloc(&workspace, workspace_size);

    // Prepare input and output pointers
    void* fwd_input_ptrs[2] = {d_input, d_filter};
    void* fwd_output_ptrs[1] = {d_output};

    // Execute forward graph
    status = execute_graph(handle, fwd_graph, fwd_input_ptrs, fwd_output_ptrs, workspace);
    cudaDeviceSynchronize();
    if (status != CONV_SUCCESS) {
        std::cerr << "Failed to execute forward graph." << std::endl;
    }

    // Print input and output tensors
    copy_and_print_tensor(d_input, input_size, "Input Tensor");
    copy_and_print_tensor(d_filter, filter_size, "Filter Tensor");
    copy_and_print_tensor(d_output, output_size, "Output Tensor (Forward)");

    // Clean up forward graph
    destroy_graph(fwd_graph);
    cudaFree(workspace);

    // Now perform backward data (data gradient) operation
    // Build backward data graph
    ConvGraph_t bwd_data_graph;
    status = build_dgrad_graph(handle, &bwd_data_graph, &output_desc, &filter_desc, &input_desc, &conv_desc, CONV_DATA_TYPE_FLOAT);
    if (status != CONV_SUCCESS) {
        std::cerr << "Failed to build backward data graph." << std::endl;
        return -1;
    }

    // Get workspace size
    status = get_workspace_size(bwd_data_graph, &workspace_size);
    if (status != CONV_SUCCESS) {
        std::cerr << "Failed to get workspace size for backward data graph." << std::endl;
        destroy_graph(bwd_data_graph);
        return -1;
    }

    cudaMalloc(&workspace, workspace_size);

    // Allocate memory for gradient input (d_output) and gradient output (d_input_grad)
    float* d_input_grad;
    cudaMalloc(&d_input_grad, input_size * sizeof(float));

    // For simplicity, let's assume d_output is filled with ones
    cudaMemset(d_output, 0, output_size * sizeof(float));
    cudaMemset(d_input_grad, 0, input_size * sizeof(float));
    cudaMemcpy(d_output, h_input.data(), output_size * sizeof(float), cudaMemcpyHostToDevice); // Using h_input as dummy data

    // Prepare input and output pointers
    void* bwd_data_input_ptrs[2] = {d_output, d_filter}; // dY and W
    void* bwd_data_output_ptrs[1] = {d_input_grad};      // dX

    // Execute backward data graph
    status = execute_graph(handle, bwd_data_graph, bwd_data_input_ptrs, bwd_data_output_ptrs, workspace);
    cudaDeviceSynchronize();
    if (status != CONV_SUCCESS) {
        std::cerr << "Failed to execute backward data graph." << std::endl;
    }

    // Print input gradients
    copy_and_print_tensor(d_input_grad, input_size, "Input Tensor Gradient (Backward Data)");

    // Clean up backward data graph
    destroy_graph(bwd_data_graph);
    cudaFree(workspace);

    // Now perform backward filter (weight gradient) operation
    // Build backward filter graph
    ConvGraph_t bwd_filter_graph;
    status = build_wgrad_graph(handle, &bwd_filter_graph, &input_desc, &output_desc, &filter_desc, &conv_desc, CONV_DATA_TYPE_FLOAT);
    if (status != CONV_SUCCESS) {
        std::cerr << "Failed to build backward filter graph." << std::endl;
        return -1;
    }

    // Get workspace size
    status = get_workspace_size(bwd_filter_graph, &workspace_size);
    if (status != CONV_SUCCESS) {
        std::cerr << "Failed to get workspace size for backward filter graph." << std::endl;
        destroy_graph(bwd_filter_graph);
        return -1;
    }

    cudaMalloc(&workspace, workspace_size);

    // Allocate memory for gradient output (d_filter_grad)
    float* d_filter_grad;
    cudaMalloc(&d_filter_grad, filter_size * sizeof(float));

    // Prepare input and output pointers
    void* bwd_filter_input_ptrs[2] = {d_input, d_output}; // X and dY
    void* bwd_filter_output_ptrs[1] = {d_filter_grad};    // dW

    // Execute backward filter graph
    status = execute_graph(handle, bwd_filter_graph, bwd_filter_input_ptrs, bwd_filter_output_ptrs, workspace);
    cudaDeviceSynchronize();
    if (status != CONV_SUCCESS) {
        std::cerr << "Failed to execute backward filter graph." << std::endl;
    }

    // Print filter gradients
    copy_and_print_tensor(d_filter_grad, filter_size, "Filter Tensor Gradient (Backward Filter)");

    // Clean up backward filter graph
    destroy_graph(bwd_filter_graph);
    cudaFree(workspace);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(d_input_grad);
    cudaFree(d_filter_grad);

    cudnnDestroy(handle);

    return 0;
}

// Include the definitions of print_tensor and copy_and_print_tensor
void print_tensor(const float* tensor, int size, const std::string& name) {
    std::cout << name << ": [";
    for (int i = 0; i < size; ++i) {
        std::cout << tensor[i];
        if (i < size - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void copy_and_print_tensor(float* device_tensor, int size, const std::string& name) {
    std::vector<float> host_tensor(size);
    cudaMemcpy(host_tensor.data(), device_tensor, size * sizeof(float), cudaMemcpyDeviceToHost);
    print_tensor(host_tensor.data(), size, name);
}

