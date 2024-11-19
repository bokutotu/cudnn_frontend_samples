#include "conv.h"
#include <cudnn_frontend.h>
#include <memory>

struct GraphComponents {
    cudnn_frontend::graph::Graph graph;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> DY;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> W;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> DX;
};

std::pair<std::vector<long int>, std::vector<long int>> to_vector(Shape* shape) {
    return std::make_pair(
        std::vector<long int>(shape->dim, shape->dim + shape->size),
        std::vector<long int>(shape->stride, shape->stride + shape->size));
}

std::tuple<std::vector<long int>, std::vector<long int>, std::vector<long int>> to_tuple(ConvParams* params) {
    return std::make_tuple(
        std::vector<long int>(params->padding, params->padding + params->size),
        std::vector<long int>(params->stride, params->stride + params->size),
        std::vector<long int>(params->dilation, params->dilation + params->size));
}

GraphComponentsHandle create_graph(
    Shape* input_shape,
    Shape* weight_shape,
    Shape* output_shape,
    ConvParams* conv_params) {

    auto [input_dim, input_stride] = to_vector(input_shape);
    auto [weight_dim, weight_stride] = to_vector(weight_shape);
    auto [output_dim, output_stride] = to_vector(output_shape);
    auto [padding, stride, dilation] = to_tuple(conv_params);

    namespace fe = cudnn_frontend;
    auto components = new GraphComponents();

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

    auto dgrad_options = fe::graph::Conv_dgrad_attributes()
                             .set_padding(padding)
                             .set_stride(stride)
                             .set_dilation(dilation);
    auto DX            = graph.conv_dgrad(DY, W, dgrad_options);
    DX->set_dim(output_dim)
       .set_stride(output_stride)
       .set_output(true);

    components->graph = graph;
    components->DY = DY;
    components->W = W;
    components->DX = DX;

    return static_cast<GraphComponentsHandle>(components);
}

bool execute_graph(
    cudnnHandle_t handle,
    GraphComponentsHandle components_handle,
    float* dy_tensor,
    float* w_tensor,
    float* dx_tensor) {

    auto components = static_cast<GraphComponents*>(components_handle);
    auto& graph = components->graph;

    if (!graph.validate().is_good()) {
        std::cerr << "Graph validation failed" << std::endl;
        return false;
    }

    if (!graph.build_operation_graph(handle).is_good()) {
        std::cerr << "Failed to build operation graph" << std::endl;
        return false;
    }

    if (!graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good()) {
        std::cerr << "Failed to create execution plans" << std::endl;
        return false;
    }

    if (!graph.check_support(handle).is_good()) {
        std::cerr << "Graph not supported" << std::endl;
        return false;
    }

    if (!graph.build_plans(handle).is_good()) {
        std::cerr << "Failed to build plans" << std::endl;
        return false;
    }

    int64_t workspace_size;
    if (!graph.get_workspace_size(workspace_size).is_good()) {
        std::cerr << "Failed to get workspace size" << std::endl;
        return false;
    }

    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> variant_pack = {
        {components->DY, dy_tensor},
        {components->W, w_tensor},
        {components->DX, dx_tensor}
    };

    bool success = graph.execute(handle, variant_pack, workspace).is_good();

    cudaFree(workspace);
    return success;
}

void destroy_graph_components(GraphComponentsHandle components_handle) {
    auto components = static_cast<GraphComponents*>(components_handle);
    delete components;
}
