#pragma once

#include "../include/cudnn_frontend_wrapper.h"
#include "cudnn_frontend.h"

namespace fe = cudnn_frontend;

struct BatchNormTensorAttributes {
    BatchNormTensorAttributes() = default;

    void debug_print();

    BatchNormTensorAttributes(std::shared_ptr<fe::graph::Tensor_attributes> X,
                              std::shared_ptr<fe::graph::Tensor_attributes> prev_running_mean,
                              std::shared_ptr<fe::graph::Tensor_attributes> prev_running_var,
                              std::shared_ptr<fe::graph::Tensor_attributes> scale,
                              std::shared_ptr<fe::graph::Tensor_attributes> bias,
                              std::shared_ptr<fe::graph::Tensor_attributes> peer_stats_0,
                              std::shared_ptr<fe::graph::Tensor_attributes> peer_stats_1,
                              std::shared_ptr<fe::graph::Tensor_attributes> epsilon,
                              std::shared_ptr<fe::graph::Tensor_attributes> momentum,
                              std::shared_ptr<fe::graph::Tensor_attributes> next_running_mean,
                              std::shared_ptr<fe::graph::Tensor_attributes> next_running_var,
                              std::shared_ptr<fe::graph::Tensor_attributes> bn_output,
                              std::shared_ptr<fe::graph::Tensor_attributes> mean,
                              std::shared_ptr<fe::graph::Tensor_attributes> inv_variance) :
        X(X),
        prev_running_mean(prev_running_mean),
        prev_running_var(prev_running_var),
        scale(scale),
        bias(bias),
        peer_stats_0(peer_stats_0),
        peer_stats_1(peer_stats_1),
        epsilon(epsilon),
        momentum(momentum),
        next_running_mean(next_running_mean),
        next_running_var(next_running_var),
        mean(mean),
        inv_variance(inv_variance),
        bn_output(bn_output)
        {}

    BatchNormTensorAttributes(CudnnTensorShapeStride input_shape, 
                              fe::graph::Graph &graph, 
                              CudnnFrontendDataType_t type, 
                              bool has_running_stats,
                              float epsilon,
                              float momentum);

    std::shared_ptr<fe::graph::Tensor_attributes> X;
    std::shared_ptr<fe::graph::Tensor_attributes> prev_running_mean;
    std::shared_ptr<fe::graph::Tensor_attributes> prev_running_var;
    std::shared_ptr<fe::graph::Tensor_attributes> scale;
    std::shared_ptr<fe::graph::Tensor_attributes> bias;
    std::shared_ptr<fe::graph::Tensor_attributes> peer_stats_0;
    std::shared_ptr<fe::graph::Tensor_attributes> peer_stats_1;
    std::shared_ptr<fe::graph::Tensor_attributes> epsilon;
    std::shared_ptr<fe::graph::Tensor_attributes> momentum;
    std::shared_ptr<fe::graph::Tensor_attributes> next_running_mean;
    std::shared_ptr<fe::graph::Tensor_attributes> next_running_var;
    std::shared_ptr<fe::graph::Tensor_attributes> bn_output;
    std::shared_ptr<fe::graph::Tensor_attributes> mean;
    std::shared_ptr<fe::graph::Tensor_attributes> inv_variance;
};

struct BatchNormDescriptor {
private:
    fe::graph::Graph graph;
    BatchNormTensorAttributes attributes;
    bool has_running_stats;

public:
    fe::graph::Graph get_graph() { return graph; }

    BatchNormDescriptor(fe::graph::Graph graph, BatchNormTensorAttributes attributes, bool has_running_stats) :
        graph(graph),
        attributes(attributes),
        has_running_stats(has_running_stats)
        {}

    BatchNormDescriptor(CudnnTensorShapeStride input_shape_stride, 
                        bool has_running_stats, 
                        CudnnFrontendDataType_t type, 
                        float epsilon, 
                        float momentum);

    CudnnFrontendError_t check_graph(cudnnHandle_t* handle);

    CudnnFrontendError_t get_workspace_size(int64_t* workspace_size);

    void debug_print() {
        attributes.debug_print();
    }

    CudnnFrontendError_t execute(cudnnHandle_t* handle, BatchNormExecutionBuffers* buffers, void* workspace);
};
