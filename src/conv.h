#pragma once

#include "../include/cudnn_frontend_wrapper.h"
#include  "i_graph_desc.h"
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

class ConvAttributes {
public:
    std::shared_ptr<fe::graph::Tensor_attributes> X;
    std::shared_ptr<fe::graph::Tensor_attributes> W;
    std::shared_ptr<fe::graph::Tensor_attributes> Y;

    ConvAttributes() = default;

    ConvAttributes(CudnnTensorShapeStride* x_shape, 
                   CudnnTensorShapeStride* w_shape, 
                   CudnnTensorShapeStride* y_shape, 
                   fe::graph::Graph& graph, 
                   CudnnFrontendDataType_t type,
                   ConvInfo* info);
};

struct ConvDescriptor : public IGraphDescriptor {
protected:
    fe::graph::Graph graph;
    std::vector<fe::HeurMode_t> heur_mode = {fe::HeurMode_t::A};

private:
    ConvAttributes attributes;

public:
    ConvDescriptor(CudnnFrontendDataType_t type, 
              CudnnTensorShapeStride* x_shape, 
              CudnnTensorShapeStride* w_shape, 
              CudnnTensorShapeStride* y_shape, 
              ConvInfo* info);

    CudnnFrontendError_t execute(cudnnHandle_t* handle, ConvBufers* buffers, void* workspace);
};
