#include "conv.h"
#include <cudnn_frontend.h>
#include <memory>
#include <array>

// Define unique IDs for tensors
constexpr long int X_TENSOR_ID = 0;
constexpr long int W_TENSOR_ID = 1;
constexpr long int Y_TENSOR_ID = 2;
constexpr long int DX_TENSOR_ID = 3;
constexpr long int DY_TENSOR_ID = 4;
constexpr long int DW_TENSOR_ID = 5;

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

struct GraphComponents {
    std::unique_ptr<cudnn_frontend::OperationGraph> graph;
    std::unique_ptr<cudnn_frontend::Tensor> tensor_a;
    std::unique_ptr<cudnn_frontend::Tensor> tensor_b;
    std::unique_ptr<cudnn_frontend::Tensor> tensor_c;
};

GraphComponentsHandle create_forward_graph(
    cudnnHandle_t handle,
    Shape* input_shape,
    Shape* weight_shape,
    Shape* output_shape,
    ConvParams* conv_params) {

    auto components = new GraphComponents();

    auto [x_dim, x_stride] = to_vector(input_shape);
    auto [w_dim, w_stride] = to_vector(weight_shape);
    auto [y_dim, y_stride] = to_vector(output_shape);
    auto [padding, conv_stride, dilation] = to_tuple(conv_params);

    // Create tensors
    auto tensor_x = cudnn_frontend::TensorBuilder()
                        .setDim(x_dim.size(), x_dim.data())
                        .setStrides(x_stride.size(), x_stride.data())
                        .setId(X_TENSOR_ID)
                        .setAlignment(16)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();

    auto tensor_w = cudnn_frontend::TensorBuilder()
                        .setDim(w_dim.size(), w_dim.data())
                        .setStrides(w_stride.size(), w_stride.data())
                        .setId(W_TENSOR_ID)
                        .setAlignment(16)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();

    auto tensor_y = cudnn_frontend::TensorBuilder()
                        .setDim(y_dim.size(), y_dim.data())
                        .setStrides(y_stride.size(), y_stride.data())
                        .setId(Y_TENSOR_ID)
                        .setAlignment(16)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();

    auto convDesc = cudnn_frontend::ConvDescBuilder()
                        .setDataType(CUDNN_DATA_FLOAT)
                        .setMathMode(CUDNN_CROSS_CORRELATION)
                        .setNDims(conv_params->size)
                        .setStrides(conv_params->size, conv_stride.data())
                        .setPrePadding(conv_params->size, padding.data())
                        .setPostPadding(conv_params->size, padding.data())
                        .setDilation(conv_params->size, dilation.data())
                        .build();

    // Convolution operation
    auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                       .setxDesc(tensor_x)
                       .setwDesc(tensor_w)
                       .setyDesc(tensor_y)
                       .setcDesc(convDesc)
                       .setAlpha(1.0f)
                       .setBeta(0.0f)
                       .build();

    // Build the graph
    std::array<cudnn_frontend::Operation const*, 1> ops = {&conv_op};
    auto opGraph = cudnn_frontend::OperationGraphBuilder()
                       .setHandle(handle)
                       .setOperationGraph(ops.size(), ops.data())
                       .build();

    components->graph = std::make_unique<cudnn_frontend::OperationGraph>(std::move(opGraph));
    components->tensor_a = std::make_unique<cudnn_frontend::Tensor>(std::move(tensor_x));
    components->tensor_b = std::make_unique<cudnn_frontend::Tensor>(std::move(tensor_w));
    components->tensor_c = std::make_unique<cudnn_frontend::Tensor>(std::move(tensor_y));

    return static_cast<GraphComponentsHandle>(components);
}

GraphComponentsHandle create_backward_data_graph(
    cudnnHandle_t handle,
    Shape* input_shape,
    Shape* weight_shape,
    Shape* output_shape,
    ConvParams* conv_params) {
    auto components = new GraphComponents();

    auto [x_dim, x_stride] = to_vector(input_shape);
    auto [w_dim, w_stride] = to_vector(weight_shape);
    auto [y_dim, y_stride] = to_vector(output_shape);
    auto [padding, conv_stride, dilation] = to_tuple(conv_params);

    // Create tensors
    auto tensor_dy = cudnn_frontend::TensorBuilder()
                         .setDim(y_dim.size(), y_dim.data())
                         .setStrides(y_stride.size(), y_stride.data())
                         .setId(DY_TENSOR_ID)
                         .setAlignment(16)
                         .setDataType(CUDNN_DATA_FLOAT)
                         .build();

    auto tensor_w = cudnn_frontend::TensorBuilder()
                        .setDim(w_dim.size(), w_dim.data())
                        .setStrides(w_stride.size(), w_stride.data())
                        .setId(W_TENSOR_ID)
                        .setAlignment(16)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();

    auto tensor_dx = cudnn_frontend::TensorBuilder()
                         .setDim(x_dim.size(), x_dim.data())
                         .setStrides(x_stride.size(), x_stride.data())
                         .setId(DX_TENSOR_ID)
                         .setAlignment(16)
                         .setDataType(CUDNN_DATA_FLOAT)
                         .build();

    auto convDesc = cudnn_frontend::ConvDescBuilder()
                        .setDataType(CUDNN_DATA_FLOAT)
                        .setMathMode(CUDNN_CROSS_CORRELATION)
                        .setNDims(conv_params->size)
                        .setStrides(conv_params->size, conv_stride.data())
                        .setPrePadding(conv_params->size, padding.data())
                        .setPostPadding(conv_params->size, padding.data())
                        .setDilation(conv_params->size, dilation.data())
                        .build();

    // Backward data operation
    auto backward_data_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                                .setdyDesc(tensor_dy)
                                .setwDesc(tensor_w)
                                .setdxDesc(tensor_dx)
                                .setcDesc(convDesc)
                                .setAlpha(1.0f)
                                .setBeta(0.0f)
                                .build();

    // Build the graph
    std::array<cudnn_frontend::Operation const*, 1> ops = {&backward_data_op};
    auto opGraph = cudnn_frontend::OperationGraphBuilder()
                       .setHandle(handle)
                       .setOperationGraph(ops.size(), ops.data())
                       .build();

    components->graph = std::make_unique<cudnn_frontend::OperationGraph>(std::move(opGraph));
    components->tensor_a = std::make_unique<cudnn_frontend::Tensor>(std::move(tensor_dy));
    components->tensor_b = std::make_unique<cudnn_frontend::Tensor>(std::move(tensor_w));
    components->tensor_c = std::make_unique<cudnn_frontend::Tensor>(std::move(tensor_dx));

    return static_cast<GraphComponentsHandle>(components);
}

GraphComponentsHandle create_backward_filter_graph(
    cudnnHandle_t handle,
    Shape* input_shape,
    Shape* weight_shape,
    Shape* output_shape,
    ConvParams* conv_params) {
    auto components = new GraphComponents();

    auto [x_dim, x_stride] = to_vector(input_shape);
    auto [w_dim, w_stride] = to_vector(weight_shape);
    auto [y_dim, y_stride] = to_vector(output_shape);
    auto [padding, conv_stride, dilation] = to_tuple(conv_params);

    // Create tensors
    auto tensor_x = cudnn_frontend::TensorBuilder()
                        .setDim(x_dim.size(), x_dim.data())
                        .setStrides(x_stride.size(), x_stride.data())
                        .setId(X_TENSOR_ID)
                        .setAlignment(16)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();

    auto tensor_dy = cudnn_frontend::TensorBuilder()
                         .setDim(y_dim.size(), y_dim.data())
                         .setStrides(y_stride.size(), y_stride.data())
                         .setId(DY_TENSOR_ID)
                         .setAlignment(16)
                         .setDataType(CUDNN_DATA_FLOAT)
                         .build();

    auto tensor_dw = cudnn_frontend::TensorBuilder()
                         .setDim(w_dim.size(), w_dim.data())
                         .setStrides(w_stride.size(), w_stride.data())
                         .setId(DW_TENSOR_ID)
                         .setAlignment(16)
                         .setDataType(CUDNN_DATA_FLOAT)
                         .build();

    auto convDesc = cudnn_frontend::ConvDescBuilder()
                        .setDataType(CUDNN_DATA_FLOAT)
                        .setMathMode(CUDNN_CROSS_CORRELATION)
                        .setNDims(conv_params->size)
                        .setStrides(conv_params->size, conv_stride.data())
                        .setPrePadding(conv_params->size, padding.data())
                        .setPostPadding(conv_params->size, padding.data())
                        .setDilation(conv_params->size, dilation.data())
                        .build();

    // Backward filter operation
    auto backward_filter_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
                                  .setxDesc(tensor_x)
                                  .setdyDesc(tensor_dy)
                                  .setwDesc(tensor_dw)
                                  .setcDesc(convDesc)
                                  .setAlpha(1.0f)
                                  .setBeta(0.0f)
                                  .build();

    // Build the graph
    std::array<cudnn_frontend::Operation const*, 1> ops = {&backward_filter_op};
    auto opGraph = cudnn_frontend::OperationGraphBuilder()
                       .setHandle(handle)
                       .setOperationGraph(ops.size(), ops.data())
                       .build();

    components->graph = std::make_unique<cudnn_frontend::OperationGraph>(std::move(opGraph));
    components->tensor_a = std::make_unique<cudnn_frontend::Tensor>(std::move(tensor_x));
    components->tensor_b = std::make_unique<cudnn_frontend::Tensor>(std::move(tensor_dy));
    components->tensor_c = std::make_unique<cudnn_frontend::Tensor>(std::move(tensor_dw));

    return static_cast<GraphComponentsHandle>(components);
}

bool execute_graph(
    cudnnHandle_t handle,
    GraphComponentsHandle components_handle,
    void* tensor_a,
    void* tensor_b,
    void* tensor_c) {

    auto components = static_cast<GraphComponents*>(components_handle);

    // Select the engine
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                          .setOperationGraph(*(components->graph))
                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                          .build();

    auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

    // Create the execution plan
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(engine_configs[0], components->graph->getTag())
                    .build();

    if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Failed to create execution plan" << std::endl;
        return false;
    }

    // Prepare the variant pack
    void* data_ptrs[] = {tensor_a, tensor_b, tensor_c};
    int64_t uids[]    = {
        components->tensor_a->getId(),
        components->tensor_b->getId(),
        components->tensor_c->getId()
    };

    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(nullptr)
                           .setDataPointers(3, data_ptrs)
                           .setUids(3, uids)
                           .build();

    if (variantPack.get_status() != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Failed to create variant pack" << std::endl;
        return false;
    }

    // Execute
    auto status = cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Execution failed" << std::endl;
        return false;
    }

    return true;
}

void destroy_graph_components(GraphComponentsHandle components_handle) {
    auto components = static_cast<GraphComponents*>(components_handle);
    delete components;
}

