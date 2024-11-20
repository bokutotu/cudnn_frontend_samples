// #include "conv.h"
// #include <cudnn_frontend.h>
// #include <memory>
//
// struct GraphComponents {
//     cudnn_frontend::graph::Graph graph;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> DY;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> W;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> DX;
// };
//
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
//
// GraphComponentsHandle create_graph(
//     Shape* input_shape,
//     Shape* weight_shape,
//     Shape* output_shape,
//     ConvParams* conv_params) {
//
//     auto [input_dim, input_stride] = to_vector(input_shape);
//     auto [weight_dim, weight_stride] = to_vector(weight_shape);
//     auto [output_dim, output_stride] = to_vector(output_shape);
//     auto [padding, stride, dilation] = to_tuple(conv_params);
//
//     namespace fe = cudnn_frontend;
//     auto components = new GraphComponents();
//
//     fe::graph::Graph graph;
//     graph.set_io_data_type(fe::DataType_t::FLOAT)
//          .set_intermediate_data_type(fe::DataType_t::FLOAT)
//          .set_compute_data_type(fe::DataType_t::FLOAT);
//
//     auto DY = graph.tensor(fe::graph::Tensor_attributes()
//                                .set_name("grad")
//                                .set_dim(input_dim)
//                                .set_stride(input_stride));
//     auto W  = graph.tensor(fe::graph::Tensor_attributes()
//                                .set_name("weight")
//                                .set_dim(weight_dim)
//                                .set_stride(weight_stride));
//
//     auto dgrad_options = fe::graph::Conv_dgrad_attributes()
//                              .set_padding(padding)
//                              .set_stride(stride)
//                              .set_dilation(dilation);
//     auto DX            = graph.conv_dgrad(DY, W, dgrad_options);
//     DX->set_dim(output_dim)
//        .set_stride(output_stride)
//        .set_output(true);
//
//     components->graph = graph;
//     components->DY = DY;
//     components->W = W;
//     components->DX = DX;
//
//     return static_cast<GraphComponentsHandle>(components);
// }
//
// bool execute_graph(
//     cudnnHandle_t handle,
//     GraphComponentsHandle components_handle,
//     float* dy_tensor,
//     float* w_tensor,
//     float* dx_tensor) {
//
//     auto components = static_cast<GraphComponents*>(components_handle);
//     auto& graph = components->graph;
//
//     if (!graph.validate().is_good()) {
//         std::cerr << "Graph validation failed" << std::endl;
//         return false;
//     }
//
//     if (!graph.build_operation_graph(handle).is_good()) {
//         std::cerr << "Failed to build operation graph" << std::endl;
//         return false;
//     }
//
//     if (!graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good()) {
//         std::cerr << "Failed to create execution plans" << std::endl;
//         return false;
//     }
//
//     if (!graph.check_support(handle).is_good()) {
//         std::cerr << "Graph not supported" << std::endl;
//         return false;
//     }
//
//     if (!graph.build_plans(handle).is_good()) {
//         std::cerr << "Failed to build plans" << std::endl;
//         return false;
//     }
//
//     int64_t workspace_size;
//     if (!graph.get_workspace_size(workspace_size).is_good()) {
//         std::cerr << "Failed to get workspace size" << std::endl;
//         return false;
//     }
//
//     void* workspace = nullptr;
//     cudaMalloc(&workspace, workspace_size);
//
//     std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> variant_pack = {
//         {components->DY, dy_tensor},
//         {components->W, w_tensor},
//         {components->DX, dx_tensor}
//     };
//
//     bool success = graph.execute(handle, variant_pack, workspace).is_good();
//
//     cudaFree(workspace);
//     return success;
// }
//
// void destroy_graph_components(GraphComponentsHandle components_handle) {
//     auto components = static_cast<GraphComponents*>(components_handle);
//     delete components;
// }
#include "conv.h"
#include <cudnn_frontend.h>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <cuda_runtime.h>

struct GraphComponents {
    cudnn_frontend::Graph graph;
    std::shared_ptr<cudnn_frontend::Tensor> tensor_a;
    std::shared_ptr<cudnn_frontend::Tensor> tensor_b;
    std::shared_ptr<cudnn_frontend::Tensor> tensor_c;
};

GraphComponentsHandle create_forward_graph_c(
    Shape* input_shape,
    Shape* weight_shape,
    Shape* output_shape,
    ConvParams* conv_params) {

    auto components = new GraphComponents();

    // // 構築処理
    // std::vector<int64_t> x_dim(input_shape->dim, input_shape->dim + input_shape->size);
    // std::vector<int64_t> x_stride(input_shape->stride, input_shape->stride + input_shape->size);
    // std::vector<int64_t> w_dim(weight_shape->dim, weight_shape->dim + weight_shape->size);
    // std::vector<int64_t> w_stride(weight_shape->stride, weight_shape->stride + weight_shape->size);
    // std::vector<int64_t> y_dim(output_shape->dim, output_shape->dim + output_shape->size);
    // std::vector<int64_t> y_stride(output_shape->stride, output_shape->stride + output_shape->size);
    // std::vector<int64_t> padding(conv_params->padding, conv_params->padding + conv_params->size);
    // std::vector<int64_t> conv_stride(conv_params->stride, conv_params->stride + conv_params->size);
    // std::vector<int64_t> dilation(conv_params->dilation, conv_params->dilation + conv_params->size);
    auto [x_dim, x_stride] = to_vector(input_shape);
    auto [w_dim, w_stride] = to_vector(weight_shape);
    auto [y_dim, y_stride] = to_vector(output_shape);
    auto [padding, conv_stride, dilation] = to_tuple(conv_params);

    // テンソルの作成
    auto tensor_x = cudnn_frontend::TensorBuilder()
                        .setDim(x_dim.size(), x_dim.data())
                        .setStrides(x_stride.size(), x_stride.data())
                        .setId('x')
                        .setAlignment(16)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();

    auto tensor_w = cudnn_frontend::TensorBuilder()
                        .setDim(w_dim.size(), w_dim.data())
                        .setStrides(w_stride.size(), w_stride.data())
                        .setId('w')
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

    // 畳み込みの操作
    auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                       .setxDesc(tensor_x)
                       .setwDesc(tensor_w)
                       .setyDesc('y')
                       .setcDesc(convDesc)
                       .setAlpha(1.0f)
                       .setBeta(0.0f)
                       .build();

    // グラフの構築
    std::array<cudnn_frontend::Operation const*, 1> ops = {&conv_op};
    auto opGraph = cudnn_frontend::OperationGraphBuilder()
                       .setHandle(cudnnHandle_t())
                       .setOperationGraph(ops.size(), ops.data())
                       .build();

    components->graph = std::move(opGraph);
    components->tensor_a = std::make_shared<cudnn_frontend::Tensor>(tensor_x);
    components->tensor_b = std::make_shared<cudnn_frontend::Tensor>(tensor_w);
    // 出力テンソルは操作から取得
    components->tensor_c = std::make_shared<cudnn_frontend::Tensor>(conv_op.getOutputTensor());

    return static_cast<GraphComponentsHandle>(components);
}

GraphComponentsHandle create_backward_data_graph_c(
    Shape* input_shape,
    Shape* weight_shape,
    Shape* output_shape,
    ConvParams* conv_params) {

    auto components = new GraphComponents();

    // 同様にバックワードデータのグラフを作成
    // ...（省略：前方と同様に設定）

    auto [x_dim, x_stride] = to_vector(input_shape);
    auto [w_dim, w_stride] = to_vector(weight_shape);
    auto [y_dim, y_stride] = to_vector(output_shape);
    auto [padding, conv_stride, dilation] = to_tuple(conv_params);

    auto tensor_x = cudnn_frontend::TensorBuilder()
                        .setDim(x_dim.size(), x_dim.data())
                        .setStrides(x_stride.size(), x_stride.data())
                        .setId('x')
                        .setAlignment(16)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();

    auto tensor_w = cudnn_frontend::TensorBuilder()
                        .setDim(w_dim.size(), w_dim.data())
                        .setStrides(w_stride.size(), w_stride.data())
                        .setId('w')
                        .setAlignment(16)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();

    auto tensor_y = cudnn_frontend::TensorBuilder()
                        .setDim(y_dim.size(), y_dim.data())
                        .setStrides(y_stride.size(), y_stride.data())
                        .setId('y')
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

    auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                        .setxDesc(tensor_x)
                        .setwDesc(tensor_w)
                        .setyDesc(tensor_y)
                        .setcDesc(convDesc)
                        .setAlpha(1.0f)


    return static_cast<GraphComponentsHandle>(components);
}

GraphComponentsHandle create_backward_filter_graph_c(
    Shape* input_shape,
    Shape* weight_shape,
    Shape* output_shape,
    ConvParams* conv_params) {

    auto components = new GraphComponents();

    // 同様にバックワードフィルタのグラフを作成
    // ...（省略：前方と同様に設定）

    return static_cast<GraphComponentsHandle>(components);
}

bool execute_graph_c(
    cudnnHandle_t handle,
    GraphComponentsHandle components_handle,
    void* tensor_a,
    void* tensor_b,
    void* tensor_c) {

    auto components = static_cast<GraphComponents*>(components_handle);

    // エンジンの選択
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                          .setOperationGraph(components->graph)
                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                          .build();

    auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

    // プランの作成
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(engine_configs[0], components->graph.getTag())
                    .build();

    if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Failed to create execution plan" << std::endl;
        return false;
    }

    // バリアントパックの準備
    void* data_ptrs[] = {tensor_a, tensor_b, tensor_c};
    int64_t uids[]    = {'x', 'w', 'y'};

    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(nullptr)
                           .setDataPointers(3, data_ptrs)
                           .setUids(3, uids)
                           .build();

    if (variantPack.get_status() != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Failed to create variant pack" << std::endl;
        return false;
    }

    // 実行
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
