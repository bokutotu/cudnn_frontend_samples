#include <iostream>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "cudnn_frontend.h"
#include "helpers.h"

#define REQUIRE_(x) if (!(x)) { throw std::runtime_error("Error at line " + std::to_string(__LINE__)); }

int main() {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    bool has_running_stats = true;
    auto X                 = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({4, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));
    auto prev_running_mean = graph.tensor(fe::graph::Tensor_attributes()
                                              .set_name("prev_running_mean")
                                              .set_dim({1, 32, 1, 1})
                                              .set_stride({32, 1, 32, 32})
                                              .set_data_type(fe::DataType_t::FLOAT));
    auto prev_running_var  = graph.tensor(fe::graph::Tensor_attributes()
                                             .set_name("prev_running_var")
                                             .set_dim({1, 32, 1, 1})
                                             .set_stride({32, 1, 32, 32})
                                             .set_data_type(fe::DataType_t::FLOAT));
    auto scale             = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, 32, 1, 1})
                                  .set_stride({32, 1, 32, 32})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias              = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 32, 1, 1})
                                 .set_stride({32, 1, 32, 32})
                                 .set_data_type(fe::DataType_t::FLOAT));

    auto peer_stats_0 = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_dim({2, 4 * 32, 1, 1})
                                         .set_stride({4 * 32, 1, 4 * 32, 4 * 32})
                                         .set_data_type(fe::DataType_t::FLOAT));
    auto peer_stats_1 = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_dim({2, 4 * 32, 1, 1})
                                         .set_stride({4 * 32, 1, 4 * 32, 4 * 32})
                                         .set_data_type(fe::DataType_t::FLOAT));

    auto epsilon  = graph.tensor(1e-05f);
    auto momentum = graph.tensor(1e-01f);

    auto batchnorm_options =
        fe::graph::Batchnorm_attributes().set_epsilon(epsilon).set_peer_stats({peer_stats_0, peer_stats_1});
    if (has_running_stats) {
        batchnorm_options.set_previous_running_stats(prev_running_mean, prev_running_var, momentum);
    }

    auto [bn_output, mean, inv_variance, next_running_mean, next_running_var] =
        graph.batchnorm(X, scale, bias, batchnorm_options);
    mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    if (has_running_stats) {
        next_running_mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    }
    if (has_running_stats) {
        next_running_var->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    }

    auto A           = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({4, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                              .set_data_type(fe::DataType_t::HALF));
    auto add_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto add_output  = graph.pointwise(bn_output, A, add_options);

    auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto Y            = graph.pointwise(add_output, relu_options);
    Y->set_output(true);

#if (CUDNN_VERSION < 8700)
    SKIP("single GPU BN is not supported in cudnn versions prior to 8.7");
#endif
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("ConvBNFprop requires Ampere and up");
    }
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    REQUIRE_(graph.validate().is_good());

    REQUIRE_(graph.build_operation_graph(handle).is_good());

    REQUIRE_(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE_(graph.check_support(handle).is_good());

    REQUIRE_(graph.build_plans(handle).is_good());

    Surface<half> X_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Var_tensor(32, false);
    Surface<float> Previous_running_mean_tensor(32, false);
    Surface<float> Previous_running_var_tensor(32, false);
    Surface<float> Next_running_mean_tensor(32, false);
    Surface<float> Next_running_var_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Bias_tensor(32, false);
    Surface<half> A_tensor(4 * 32 * 16 * 16, false);
    Surface<half> Y_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Peer_stats_0_tensor(2 * 4 * 32, false, true);
    Surface<float> Peer_stats_1_tensor(2 * 4 * 32, false);

    int64_t workspace_size;
    REQUIRE_(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Var_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {A, A_tensor.devPtr},
        {Y, Y_tensor.devPtr},
        {peer_stats_0, Peer_stats_0_tensor.devPtr},
        {peer_stats_1, Peer_stats_1_tensor.devPtr}};

    if (has_running_stats) {
        variant_pack[prev_running_mean] = Previous_running_mean_tensor.devPtr;
        variant_pack[prev_running_var]  = Previous_running_var_tensor.devPtr;
        variant_pack[next_running_mean] = Next_running_mean_tensor.devPtr;
        variant_pack[next_running_var]  = Next_running_var_tensor.devPtr;
    }
    REQUIRE_(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}
//
// struct BatchNormTensorDesc {
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> X;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> prev_running_mean;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> prev_running_var;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> scale;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> bias;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> mean;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> inv_variance;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> next_running_mean;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> next_running_var;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> epsilon;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> momentum;
//     std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Y;
// };
//
// struct BatchNormBuilder {
//     cudnn_frontend::DataType_t io_data_type;
//     cudnn_frontend::DataType_t intermediate_data_type;
//     cudnn_frontend::DataType_t compute_data_type;
//
//     int input_dim_len;
//     int64_t input_dim[8];
//
//     float epsilon;
//     float momentum;
// };
//
// std::vector<int64_t> get_default_stride(int len, int64_t dim[8]) {
//     std::vector<int64_t> stride(len);
//     stride[len - 1] = 1;
//     for (int i = len - 2; i >= 0; i--) {
//         stride[i] = stride[i + 1] * dim[i + 1];
//     }
//     return stride;
// }
//
// std::vector<int64_t> get_static_shape(int len, int64_t dim[8]) {
//     if (len == 2) {
//         return {1, dim[1]};
//     } else if (len == 4) {
//         return {1, dim[1], 1, 1};
//     } else {
//         throw std::runtime_error("Unsupported dimension length");
//     }
// }
//
// std::vector<int64_t> get_static_stride(int len, int64_t dim[8]) {
//     if (len == 2) {
//         return {dim[1], 1};
//     } else if (len == 4) {
//         return {dim[1], 1, dim[1], dim[1]};
//     } else {
//         throw std::runtime_error("Unsupported dimension length");
//     }
// }
//
// std::vector<int64_t> get_input_shape_vector(int len, int64_t dim[8]) {
//     std::vector<int64_t> shape(len);
//     for (int i = 0; i < len; i++) {
//         shape[i] = dim[i];
//     }
//     return shape;
// }
//
// std::pair<BatchNormTensorDesc, cudnn_frontend::graph::Graph> create_batchnorm_graph(BatchNormBuilder builder) {
//     cudnn_frontend::graph::Graph graph;
//     graph.set_io_data_type(builder.io_data_type)
//          .set_intermediate_data_type(builder.intermediate_data_type)
//          .set_compute_data_type(builder.compute_data_type);
//
//     auto input_shape = get_input_shape_vector(builder.input_dim_len, builder.input_dim);
//     auto input_stride = get_default_stride(input_shape.size(), input_shape.data());
//
//     auto static_shape = get_static_shape(builder.input_dim_len, builder.input_dim);
//     auto static_stride = get_static_stride(builder.input_dim_len, builder.input_dim);
//
//     auto X = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
//                             .set_name("X")
//                             .set_dim(input_shape)
//                             .set_stride(input_stride)
//                             .set_data_type(builder.io_data_type));
//
//     auto prev_running_mean = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
//                                             .set_name("prev_running_mean")
//                                             .set_dim(static_shape)
//                                             .set_stride(static_stride)
//                                             .set_data_type(builder.io_data_type));
//
//     auto prev_running_var = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
//                                             .set_name("prev_running_var")
//                                             .set_dim(static_shape)
//                                             .set_stride(static_stride)
//                                             .set_data_type(builder.io_data_type));
//
//     auto scale = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
//                                 .set_name("scale")
//                                 .set_dim(static_shape)
//                                 .set_stride(static_stride)
//                                 .set_data_type(builder.io_data_type));
//
//     auto bias = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
//                                 .set_name("bias")
//                                 .set_dim(static_shape)
//                                 .set_stride(static_stride)
//                                 .set_data_type(builder.io_data_type));
//
//     auto epsilon = graph.tensor(builder.epsilon);
//     auto momentum = graph.tensor(builder.momentum);
//
//     auto batchnorm_options = cudnn_frontend::graph::Batchnorm_attributes()
//                                 .set_epsilon(epsilon)
//                                 .set_previous_running_stats(prev_running_mean, prev_running_var, momentum);
//
//     auto [Y, mean, inv_variance, next_running_mean, next_running_var] =
//         graph.batchnorm(X, scale, bias, batchnorm_options);
//     mean->set_output(true).set_data_type(builder.io_data_type);
//     inv_variance->set_output(true).set_data_type(builder.io_data_type);
//
//     next_running_mean->set_output(true).set_data_type(builder.io_data_type);
//     next_running_var->set_output(true).set_data_type(builder.io_data_type);
//
//     BatchNormTensorDesc desc = {
//         X, prev_running_mean, prev_running_var, scale, bias,
//         mean, inv_variance, next_running_mean, next_running_var, epsilon, momentum, Y
//     };
//     return {desc, graph};
// }
//
// void check_graph(cudnnHandle_t handle, cudnn_frontend::graph::Graph graph) {
//     auto err = graph.validate();
//     if (!err.is_good()) {
//         throw std::runtime_error("Error validating graph");
//     }
//
//     err = graph.build_operation_graph(handle);
//     if (!err.is_good()) {
//         throw std::runtime_error("Error building operation graph");
//     }
//
//     err = graph.create_execution_plans({cudnn_frontend::HeurMode_t::FALLBACK});
//     if (!err.is_good()) {
//         throw std::runtime_error("Error creating execution plans");
//     }
//
//     err = graph.check_support(handle);
//     if (!err.is_good()) {
//         throw std::runtime_error("Error checking support");
//     }
//
//     err = graph.build_plans(handle);
//     if (!err.is_good()) {
//         throw std::runtime_error("Error building plans");
//     }
// }
//
// int64_t get_batch_norm_workspace_size(cudnn_frontend::graph::Graph graph) {
//     int64_t workspace_size;
//     auto err = graph.get_workspace_size(workspace_size);
//     if (!err.is_good()) {
//         throw std::runtime_error("Error getting workspace size");
//     }
//     return workspace_size;
// }
//
// struct BatchNormExecutePointers {
//     void *X;
//     void *prev_running_mean;
//     void *prev_running_var;
//     void *scale;
//     void *bias;
//     void *mean;
//     void *inv_variance;
//     void *next_running_mean;
//     void *next_running_var;
//     void *epsilon;
//     void *momentum;
//     void *Y;
//     void *workspace;
// };
//
// void execute_batchnorm_graph(cudnnHandle_t handle, BatchNormTensorDesc desc, cudnn_frontend::graph::Graph graph, BatchNormExecutePointers pointers) {
//     std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void *> variant_pack = {
//         {desc.X, pointers.X},
//         {desc.prev_running_mean, pointers.prev_running_mean},
//         {desc.prev_running_var, pointers.prev_running_var},
//         {desc.scale, pointers.scale},
//         {desc.bias, pointers.bias},
//         {desc.mean, pointers.mean},
//         {desc.inv_variance, pointers.inv_variance},
//         {desc.next_running_mean, pointers.next_running_mean},
//         {desc.next_running_var, pointers.next_running_var},
//         {desc.epsilon, pointers.epsilon},
//         {desc.momentum, pointers.momentum},
//         {desc.Y, pointers.Y}
//     };
//
//     auto err = graph.execute(handle, variant_pack, pointers.workspace);
// }
//
// int main() {
//     cudnnHandle_t handle;
//     cudnnCreate(&handle);
//
//     BatchNormBuilder builder = {
//         .io_data_type = cudnn_frontend::DataType_t::FLOAT,
//         .intermediate_data_type = cudnn_frontend::DataType_t::FLOAT,
//         .compute_data_type = cudnn_frontend::DataType_t::FLOAT,
//         .input_dim_len = 4,
//         .input_dim = {4, 32, 16, 16},
//         .epsilon = 1e-5,
//         .momentum = 0.1
//     };
//
//     auto [desc, graph] = create_batchnorm_graph(builder);
//
//     try {
//         check_graph(handle, graph);
//     } catch (std::runtime_error &e) {
//         std::cerr << e.what() << std::endl;
//         return 1;
//     }
//
//     int64_t workspace_size = get_batch_norm_workspace_size(graph);
//
//     // alloc memory in gpu
//     void *X, *prev_running_mean, *prev_running_var, *scale, *bias, *mean, *inv_variance, *next_running_mean, *next_running_var, *epsilon, *momentum, *Y, *workspace;
//     cudaMalloc((void**)&X, 4 * 32 * 16 * 16 * sizeof(float));
//     cudaMalloc((void**)&prev_running_mean, 32 * sizeof(float));
//     cudaMalloc((void**)&prev_running_var, 32 * sizeof(float));
//     cudaMalloc((void**)&scale, 32 * sizeof(float));
//     cudaMalloc((void**)&bias, 32 * sizeof(float));
//     cudaMalloc((void**)&mean, 32 * sizeof(float));
//     cudaMalloc((void**)&inv_variance, 32 * sizeof(float));
//     cudaMalloc((void**)&next_running_mean, 32 * sizeof(float));
//     cudaMalloc((void**)&next_running_var, 32 * sizeof(float));
//     cudaMalloc((void**)&epsilon, sizeof(float));
//     cudaMalloc((void**)&momentum, sizeof(float));
//     cudaMalloc((void**)&Y, 4 * 32 * 16 * 16 * sizeof(float));
//     cudaMalloc((void**)&workspace, workspace_size);
//
//     BatchNormExecutePointers pointers = {
//         .X = X,
//         .prev_running_mean = prev_running_mean,
//         .prev_running_var = prev_running_var,
//         .scale = scale,
//         .bias = bias,
//         .mean = mean,
//         .inv_variance = inv_variance,
//         .next_running_mean = next_running_mean,
//         .next_running_var = next_running_var,
//         .epsilon = epsilon,
//         .momentum = momentum,
//         .Y = Y,
//         .workspace = workspace
//     };
//
//     execute_batchnorm_graph(handle, desc, graph, pointers);
//     return 0;
// }
