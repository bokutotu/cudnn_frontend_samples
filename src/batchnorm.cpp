#include "cudnn_frontend.h"
#include "batchnorm.h"
#include "utils.h"
#include <vector>

std::vector<int64_t> get_stat_shape(size_t n, int64_t dims[8]) {
    std::vector<int64_t> shape = from_shape(n, dims);
    if (shape.size() != 4) {
        throw std::runtime_error("Invalid shape for stats");
    }
    shape[0] = 1;
    shape[2] = 1;
    shape[3] = 1;
    return shape;
}

std::vector<int64_t> get_stat_stride(std::vector<int64_t> shape) {
    if (shape.size() != 4) {
        throw std::runtime_error("Invalid shape for scale/bias");
    }
    std::vector<int64_t> stride(4, 1);
    stride[0] = shape[1];
    stride[1] = 1;
    stride[2] = shape[1];
    stride[3] = shape[1];
    return stride;
}

std::vector<int64_t> get_peer_stats_shape(size_t n, int64_t dims[8]) {
    std::vector<int64_t> shape = from_shape(n, dims);
    if (shape.size() != 4) {
        throw std::runtime_error("Invalid shape for peer stats");
    }
    shape[0] = 2;
    shape[1] *= 4;
    shape[2] = 1;
    shape[3] = 1;
    return shape;
}

std::vector<int64_t> get_peer_stats_stride(std::vector<int64_t> shape) {
    if (shape.size() != 4) {
        throw std::runtime_error("Invalid shape for peer stats");
    }
    std::vector<int64_t> stride(4, 1);
    stride[0] = shape[1];
    stride[1] = 1;
    stride[2] = shape[1];
    stride[3] = shape[1];
    return stride;
}

BatchNormTensorAttributes::BatchNormTensorAttributes(CudnnTensorShapeStride input_shape, 
                                                     fe::graph::Graph graph, 
                                                     CudnnFrontendDataType_t type, 
                                                     bool has_running_stats,
                                                     float epsilon,
                                                     float momentum) {
    std::vector<int64_t> x_shape = from_shape(input_shape.num_dims, input_shape.dims);
    std::vector<int64_t> x_strides = from_shape(input_shape.num_dims, input_shape.strides);

    std::vector<int64_t> stat_shape = get_stat_shape(input_shape.num_dims, input_shape.dims);
    std::vector<int64_t> stat_strides = get_stat_stride(stat_shape);

    std::vector<int64_t> peer_stats_shape = get_peer_stats_shape(input_shape.num_dims, input_shape.dims);
    std::vector<int64_t> peer_stats_strides = get_peer_stats_stride(peer_stats_shape);

    X = graph.tensor(get_tensor_attributes(x_shape, x_strides, type));
    prev_running_mean = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    prev_running_var = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    scale = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    bias = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    peer_stats_0 = graph.tensor(get_tensor_attributes(peer_stats_shape, peer_stats_strides, type));
    peer_stats_1 = graph.tensor(get_tensor_attributes(peer_stats_shape, peer_stats_strides, type));
    this->epsilon = graph.tensor(fe::graph::Tensor_attributes(epsilon));
    this->momentum = graph.tensor(fe::graph::Tensor_attributes(momentum));
    auto batchnorm_options = fe::graph::Batchnorm_attributes().set_epsilon(this->epsilon).set_peer_stats({peer_stats_0, peer_stats_1});
    if (has_running_stats) {
        batchnorm_options.set_previous_running_stats(prev_running_mean, prev_running_var, this->momentum);
    }
    auto [bn_output, mean, inv_variance, next_running_mean, next_running_var] = graph.batchnorm(X, scale, bias, batchnorm_options);
    auto data_type = get_data_type(type);
    mean->set_output(true).set_data_type(data_type);
    inv_variance->set_output(true).set_data_type(data_type);
    if (has_running_stats) {
        next_running_mean->set_output(true).set_data_type(data_type);
        next_running_var->set_output(true).set_data_type(data_type);
    }
    bn_output->set_output(true);
    this->bn_output = bn_output;
    this->mean = mean;
    this->inv_variance = inv_variance;
    this->next_running_mean = next_running_mean;
    this->next_running_var = next_running_var;

}

BatchNormDescriptor::BatchNormDescriptor(CudnnTensorShapeStride input_shape_stride, 
                                         bool has_running_stats,
                                         CudnnFrontendDataType_t type,
                                         float epsilon,
                                         float momentum) :
    has_running_stats(has_running_stats) {
    fe::graph::Graph graph;
    auto data_type = get_data_type(type);
    graph.set_io_data_type(data_type)
        .set_intermediate_data_type(data_type)
        .set_compute_data_type(data_type);

    attributes = BatchNormTensorAttributes(input_shape_stride, graph, type, has_running_stats, epsilon, momentum);
    this->graph = graph;
}

CudnnFrontendError_t BatchNormDescriptor::check_graph(cudnnHandle_t* handle) {
    auto err = graph.validate();
    if (!err.is_good()) {
        std::cout << "Graph validation " << std::endl;
        std::cout << err.get_message() << std::endl;
        return CudnnFrontendError_t::FAILURE;
    }
    err = graph.build_operation_graph(*handle);
    if (!err.is_good()) {
        std::cout << "Graph build operation graph " << std::endl;
        std::cout << err.get_message() << std::endl;
        return CudnnFrontendError_t::FAILURE;
    }
    err = graph.create_execution_plans({fe::HeurMode_t::FALLBACK});
    if (!err.is_good()) {
        std::cout << "Graph create execution plans " << std::endl;
        std::cout << err.get_message() << std::endl;
        return CudnnFrontendError_t::FAILURE;
    }
    err = graph.check_support(*handle);
    if (!err.is_good()) {
        std::cout << "Graph check support " << std::endl;
        std::cout << err.get_message() << std::endl;
        return CudnnFrontendError_t::FAILURE;
    }
    err = graph.build_plans(*handle);
    if (!err.is_good()) {
        std::cout << "Graph build plans " << std::endl;
        std::cout << err.get_message() << std::endl;
        return CudnnFrontendError_t::FAILURE;
    }
    return CudnnFrontendError_t::SUCCESS;
}

CudnnFrontendError_t BatchNormDescriptor::get_workspace_size(int64_t* workspace_size) {
    auto err = graph.get_workspace_size(*workspace_size);
    if (err.is_good()) {
        return CudnnFrontendError_t::FAILURE;
    }
    return CudnnFrontendError_t::SUCCESS;
}

CudnnFrontendError_t BatchNormDescriptor::execute(cudnnHandle_t* handle, 
                                                  BatchNormExecutionBuffers* buffers, 
                                                  void* workspace) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {attributes.X, buffers->X},
        {attributes.mean, buffers->mean},
        {attributes.inv_variance, buffers->inv_variance},
        {attributes.scale, buffers->scale},
        {attributes.bias, buffers->bias},
        {attributes.bn_output, buffers->Y},
        {attributes.peer_stats_0, buffers->peer_stats_0},
        {attributes.peer_stats_1, buffers->peer_stats_1}};

    if (has_running_stats) {
        variant_pack[attributes.prev_running_mean] = buffers->prev_running_mean;
        variant_pack[attributes.prev_running_var]  = buffers->prev_running_var;
        variant_pack[attributes.next_running_mean] = buffers->next_running_mean;
        variant_pack[attributes.next_running_var]  = buffers->next_running_var;
    }

    auto err = graph.execute(*handle, variant_pack, workspace);
    if (!err.is_good()) {
        std::cout << "Graph execute failed" << std::endl;
        return CudnnFrontendError_t::FAILURE;
    }
    return CudnnFrontendError_t::SUCCESS;
}
