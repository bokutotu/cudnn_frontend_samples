#include <iostream>
#include <vector>
#include <memory>
#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <unordered_map>

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

struct GraphComponents {
    cudnn_frontend::graph::Graph graph;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> DY;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> W;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> DX;
};

GraphComponents create_graph(
    const std::vector<long int>& input_dim, const std::vector<long int>& weight_dim, const std::vector<long int>& output_dim,
    const std::vector<long int>& input_stride, const std::vector<long int>& weight_stride, const std::vector<long int>& output_stride,
    const std::vector<long int>& padding, const std::vector<long int>& stride, const std::vector<long int>& dilation) {

    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("grad")
                               .set_dim(input_dim)
                               .set_stride(input_stride));
    auto W  = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("weight")
                               .set_dim(weight_dim)
                               .set_stride(weight_stride));

    auto dgrad_options = fe::graph::Conv_dgrad_attributes().set_padding(padding).set_stride(stride).set_dilation(dilation);
    auto DX            = graph.conv_dgrad(DY, W, dgrad_options);
    DX->set_dim(output_dim)
       .set_stride(output_stride)
       .set_output(true);

    GraphComponents components;
    components.graph = graph;
    components.DY = DY;
    components.W = W;
    components.DX = DX;
    return components;
}

bool execute_graph(cudnnHandle_t handle, GraphComponents& components,
                   float* dy_tensor, float* w_tensor, float* dx_tensor) {
    auto& graph = components.graph;

    if (graph.validate().is_good())
        std::cout << "Graph is valid" << std::endl;

    if (graph.build_operation_graph(handle).is_good())
        std::cout << "Operation graph is built" << std::endl;

    if (graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good())
        std::cout << "Execution plan is created" << std::endl;

    if (graph.check_support(handle).is_good())
        std::cout << "Graph is supported" << std::endl;

    if (graph.build_plans(handle).is_good())
        std::cout << "Plan is built" << std::endl;

    int64_t workspace_size;
    auto err = graph.get_workspace_size(workspace_size);
    if (err.is_good())
        std::cout << "Workspace size is " << workspace_size << std::endl;
    else {
        std::cout << "Error in getting workspace size" << std::endl;
        return false;
    }

    int8_t* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> variant_pack = {
        {components.DY, dy_tensor}, {components.W, w_tensor}, {components.DX, dx_tensor}};
    if (graph.execute(handle, variant_pack, workspace).is_good()) {
        std::cout << "Execution is successful" << std::endl;
        return true;
    } else {
        std::cout << "Execution failed" << std::endl;
        return false;
    }
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

    auto components = create_graph(input_dim, weight_dim, output_dim, input_stride, weight_stride, output_stride, padding, stride, dilation);

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
    return 0;
}
