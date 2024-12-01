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
    next_running_mean = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    next_running_var = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    bn_output = graph.tensor(get_tensor_attributes(x_shape, x_strides, type));
    mean = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    inv_variance = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
}
