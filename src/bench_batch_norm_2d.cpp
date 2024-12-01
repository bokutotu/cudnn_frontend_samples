#include "cudnn_frontend.h"

#define CUDA_CHECK(status)                                                                                    \
    {                                                                                                         \
        cudaError_t err = status;                                                                             \
        if (err != cudaSuccess) {                                                                             \
            std::stringstream err_msg;                                                                        \
            err_msg << "CUDA Error: " << cudaGetErrorString(err) << " (" << err << ") at " << __FILE__ << ":" \
                    << __LINE__;                                                                              \
            throw std::runtime_error(err_msg.str());                                                          \
        }                                                                                                     \
    }

#define CUDNN_CHECK(status)                                                                                     \
    {                                                                                                           \
        cudnnStatus_t err = status;                                                                             \
        if (err != CUDNN_STATUS_SUCCESS) {                                                                      \
            std::stringstream err_msg;                                                                          \
            err_msg << "cuDNN Error: " << cudnnGetErrorString(err) << " (" << err << ") at " << __FILE__ << ":" \
                    << __LINE__;                                                                                \
            throw std::runtime_error(err_msg.str());                                                            \
        }                                                                                                       \
    }

#define REQUIRE(expr)                                                                 \
    if (!(expr)) {                                                                    \
        std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << std::endl;         \
        return 1;                                                                     \
    }

template <typename T>
struct Surface {
    T* devPtr  = NULL;
    T* hostPtr = NULL;
    int64_t n_elems = 0;

   protected:
    explicit Surface() {}

   public:
    explicit Surface(int64_t n_elems, [[maybe_unused]] bool hasRef) : n_elems(n_elems) {
        CUDA_CHECK(cudaMalloc((void**)&(devPtr), (size_t)((n_elems) * sizeof(devPtr[0]))));
        hostPtr = (T*)calloc((size_t)n_elems, sizeof(hostPtr[0]));
        CUDA_CHECK(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    explicit Surface(int64_t n_elems, [[maybe_unused]] bool hasRef, bool isInterleaved) {
        (void)isInterleaved;
        CUDA_CHECK(cudaMalloc((void**)&(devPtr), (n_elems) * sizeof(devPtr[0])));
        hostPtr = (T*)calloc(n_elems, sizeof(hostPtr[0]));
        uint32_t* temp = (uint32_t*)hostPtr;
        for (auto i = 0; i < n_elems; i = i + 2) {
            temp[i + 1] = 1u;
        }

        CUDA_CHECK(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    explicit Surface(int64_t size, [[maybe_unused]] bool hasRef, T fillValue) : n_elems(size) {
        CUDA_CHECK(cudaMalloc((void**)&(devPtr), (size) * sizeof(devPtr[0])));
        hostPtr = (T*)calloc(size, sizeof(hostPtr[0]));
        for (int i = 0; i < size; i++) {
            hostPtr[i] = fillValue;
        }
        CUDA_CHECK(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

};

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

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

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
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
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
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}
