#include <iostream>
#include <vector>
#include <memory>
#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <cuda.h>

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

int main() {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("grad")
                               .set_dim({4, 64, 16})
                               .set_stride({64 * 16, 1, 64}));
    auto W  = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("weight")
                               .set_dim({64, 32, 3})
                               .set_stride({32 * 3, 1, 32}));

    auto dgrad_options = fe::graph::Conv_dgrad_attributes().set_padding({1}).set_stride({1}).set_dilation({1});
    auto DX            = graph.conv_dgrad(DY, W, dgrad_options);
    DX->set_dim({4, 32, 16}).set_output(true);

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    if (graph.validate().is_good())
        std::cout << "Graph is valid" << std::endl;

    if (graph.build_operation_graph(handle).is_good())
        std::cout << "Operation graph is built" << std::endl;

    if (graph.create_execution_plans({fe::HeurMode_t::A}).is_good())
        std::cout << "Execution plan is created" << std::endl;

    if (graph.check_support(handle).is_good())
        std::cout << "Graph is supported" << std::endl;

    if (graph.build_plans(handle).is_good())
        std::cout << "Plan is built" << std::endl;

    float* dy_tensor = nullptr;
    cudaMalloc(&dy_tensor, 4 * 64 * 16 * sizeof(float));
    float* w_tensor = nullptr;
    cudaMalloc(&w_tensor, 64 * 32 * 3 * sizeof(float));
    float* dx_tensor = nullptr;
    cudaMalloc(&dx_tensor, 4 * 32 * 16 * sizeof(float));

    int64_t workspace_size;
    auto err = graph.get_workspace_size(workspace_size);
    if (err.is_good())
        std::cout << "Workspace size is " << workspace_size << std::endl;
    else
        std::cout << "Error in getting workspace size" << std::endl;

    int8_t* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {DY, dy_tensor}, {W, w_tensor}, {DX, dx_tensor}};
    graph.execute(handle, variant_pack, workspace).is_good();
    cudnnDestroy(handle);
    return 0;
}
