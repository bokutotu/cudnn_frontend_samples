#include "conv.h"
#include "utils.h"
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> get_conv_info(ConvInfo* info) {
    std::vector<int64_t> padding(info->padding, info->padding + info->num_dims);
    std::vector<int64_t> stride(info->stride, info->stride + info->num_dims);
    std::vector<int64_t> dilation(info->dilation, info->dilation + info->num_dims);
    return std::make_tuple(padding, stride, dilation);
}

ConvAttributes::ConvAttributes(CudnnTensorShapeStride* x_shape, 
                               CudnnTensorShapeStride* w_shape, 
                               CudnnTensorShapeStride* y_shape, 
                               fe::graph::Graph& graph, 
                               CudnnFrontendDataType_t type,
                               ConvInfo* info) {
    auto data_type = get_data_type(type);

    X = graph.tensor(get_tensor_attributes(from_shape(x_shape->num_dims, x_shape->dims), 
                                                from_shape(x_shape->num_dims, x_shape->strides), 
                                                type));

    W = graph.tensor(get_tensor_attributes(from_shape(w_shape->num_dims, w_shape->dims),
                                                from_shape(w_shape->num_dims, w_shape->strides), 
                                                type));

    auto [padding, stride, dilation] = get_conv_info(info);

    auto conv_options = fe::graph::Conv_fprop_attributes()
                            .set_padding(padding)
                            .set_stride(stride)
                            .set_dilation(dilation);

    Y = graph.conv_fprop(X, W, conv_options);
    Y->set_output(true).set_stride(from_shape(y_shape->num_dims, y_shape->strides));
}

ConvGraph::ConvGraph(CudnnFrontendDataType_t type,
                     CudnnTensorShapeStride* x_shape, 
                     CudnnTensorShapeStride* w_shape, 
                     CudnnTensorShapeStride* y_shape, 
                     ConvInfo* info) {
    fe::graph::Graph graph;
    graph.set_io_data_type(get_data_type(type))
         .set_intermediate_data_type(get_data_type(type))
         .set_compute_data_type(get_data_type(type));

    attributes = ConvAttributes(x_shape, w_shape, y_shape, graph, type, info);
}

CudnnFrontendError_t ConvGraph::execute(cudnnHandle_t* handle, ConvBufers* buffers, void* workspace) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {attributes.X, buffers->X},
        {attributes.W, buffers->filter},
        {attributes.Y, buffers->Y}
    };

    return execute_graph(handle, variant_pack, workspace);
}
