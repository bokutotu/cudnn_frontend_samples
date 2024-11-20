#include "conv.h"
#include <cudnn_frontend.h>
#include <memory>
#include <vector>
#include <unordered_map>

struct ConvGraph {
    std::shared_ptr<cudnn_frontend::graph::Graph> graph_ptr;
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> input_tensors;
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> output_tensors;
};

// Function to map custom ConvDataType_t to cudnn_frontend::DataType_t
static cudnn_frontend::DataType_t getCudnnDataType(ConvDataType_t data_type) {
    using namespace cudnn_frontend;
    switch (data_type) {
        case CONV_DATA_TYPE_HALF:
            return DataType_t::HALF;
        case CONV_DATA_TYPE_FLOAT:
            return DataType_t::FLOAT;
        case CONV_DATA_TYPE_DOUBLE:
            return DataType_t::DOUBLE;
        // Add more cases as needed
        default:
            return DataType_t::FLOAT; // Default to float
    }
}

// Build Forward Propagation Graph
ConvError_t build_fprop_graph(
    cudnnHandle_t handle,
    ConvGraph_t* graph_out,
    const ConvTensorDescriptor_t* input_desc,
    const ConvTensorDescriptor_t* filter_desc,
    const ConvTensorDescriptor_t* output_desc,
    const ConvConvolutionDescriptor_t* conv_desc,
    ConvDataType_t data_type)
{
    if (graph_out == nullptr || input_desc == nullptr || filter_desc == nullptr ||
        output_desc == nullptr || conv_desc == nullptr) {
        return CONV_INVALID_VALUE;
    }
    try {
        using namespace cudnn_frontend;
        ConvGraph* conv_graph = new ConvGraph();

        auto graph = std::make_shared<graph::Graph>();
        auto cudnn_data_type = getCudnnDataType(data_type);
        graph->set_io_data_type(cudnn_data_type).set_compute_data_type(cudnn_data_type);

        // Create input tensor X
        std::vector<int64_t> x_dims(input_desc->dims, input_desc->dims + input_desc->num_dims);
        std::vector<int64_t> x_strides(input_desc->strides, input_desc->strides + input_desc->num_dims);
        auto X = graph->tensor(graph::Tensor_attributes()
                                   .set_name("input")
                                   .set_dim(x_dims)
                                   .set_stride(x_strides));

        // Create filter tensor W
        std::vector<int64_t> w_dims(filter_desc->dims, filter_desc->dims + filter_desc->num_dims);
        std::vector<int64_t> w_strides(filter_desc->strides, filter_desc->strides + filter_desc->num_dims);
        auto W = graph->tensor(graph::Tensor_attributes()
                                   .set_name("filter")
                                   .set_dim(w_dims)
                                   .set_stride(w_strides));

        // Set convolution options
        std::vector<int64_t> padding(conv_desc->padding, conv_desc->padding + conv_desc->num_dims);
        std::vector<int64_t> stride(conv_desc->stride, conv_desc->stride + conv_desc->num_dims);
        std::vector<int64_t> dilation(conv_desc->dilation, conv_desc->dilation + conv_desc->num_dims);

        auto conv_options = graph::Conv_fprop_attributes()
                                .set_padding(padding)
                                .set_stride(stride)
                                .set_dilation(dilation);

        auto Y = graph->conv_fprop(X, W, conv_options);

        // Set output tensor Y properties
        std::vector<int64_t> y_dims(output_desc->dims, output_desc->dims + output_desc->num_dims);
        Y->set_dim(y_dims).set_output(true);

        // Validate and build the graph
        auto status = graph->validate();
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->build_operation_graph(handle);
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->create_execution_plans({HeurMode_t::A});
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->check_support(handle);
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->build_plans(handle);
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        conv_graph->graph_ptr = graph;
        conv_graph->input_tensors.push_back(X);
        conv_graph->input_tensors.push_back(W);
        conv_graph->output_tensors.push_back(Y);

        *graph_out = static_cast<ConvGraph_t>(conv_graph);
        return CONV_SUCCESS;
    } catch (const std::exception& e) {
        return CONV_FAILURE;
    }
}

// Build Data Gradient Graph
ConvError_t build_dgrad_graph(
    cudnnHandle_t handle,
    ConvGraph_t* graph_out,
    const ConvTensorDescriptor_t* dy_desc,
    const ConvTensorDescriptor_t* w_desc,
    const ConvTensorDescriptor_t* dx_desc,
    const ConvConvolutionDescriptor_t* conv_desc,
    ConvDataType_t data_type)
{
    if (graph_out == nullptr || dy_desc == nullptr || w_desc == nullptr ||
        dx_desc == nullptr || conv_desc == nullptr) {
        return CONV_INVALID_VALUE;
    }
    try {
        using namespace cudnn_frontend;
        ConvGraph* conv_graph = new ConvGraph();

        auto graph = std::make_shared<graph::Graph>();
        auto cudnn_data_type = getCudnnDataType(data_type);
        graph->set_io_data_type(cudnn_data_type)
            .set_intermediate_data_type(cudnn_data_type)
            .set_compute_data_type(cudnn_data_type);

        // Create input tensor DY
        std::vector<int64_t> dy_dims(dy_desc->dims, dy_desc->dims + dy_desc->num_dims);
        std::vector<int64_t> dy_strides(dy_desc->strides, dy_desc->strides + dy_desc->num_dims);
        auto DY = graph->tensor(graph::Tensor_attributes()
                                    .set_name("grad_output")
                                    .set_dim(dy_dims)
                                    .set_stride(dy_strides));

        // Create weight tensor W
        std::vector<int64_t> w_dims(w_desc->dims, w_desc->dims + w_desc->num_dims);
        std::vector<int64_t> w_strides(w_desc->strides, w_desc->strides + w_desc->num_dims);
        auto W = graph->tensor(graph::Tensor_attributes()
                                   .set_name("weight")
                                   .set_dim(w_dims)
                                   .set_stride(w_strides));

        // Set convolution options
        std::vector<int64_t> padding(conv_desc->padding, conv_desc->padding + conv_desc->num_dims);
        std::vector<int64_t> stride(conv_desc->stride, conv_desc->stride + conv_desc->num_dims);
        std::vector<int64_t> dilation(conv_desc->dilation, conv_desc->dilation + conv_desc->num_dims);

        auto dgrad_options = graph::Conv_dgrad_attributes()
                                 .set_padding(padding)
                                 .set_stride(stride)
                                 .set_dilation(dilation);

        auto DX = graph->conv_dgrad(DY, W, dgrad_options);

        // Set output tensor DX properties
        std::vector<int64_t> dx_dims(dx_desc->dims, dx_desc->dims + dx_desc->num_dims);
        DX->set_dim(dx_dims).set_output(true);

        // Validate and build the graph
        auto status = graph->validate();
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->build_operation_graph(handle);
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->create_execution_plans({HeurMode_t::A});
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->check_support(handle);
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->build_plans(handle);
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        conv_graph->graph_ptr = graph;
        conv_graph->input_tensors.push_back(DY);
        conv_graph->input_tensors.push_back(W);
        conv_graph->output_tensors.push_back(DX);

        *graph_out = static_cast<ConvGraph_t>(conv_graph);
        return CONV_SUCCESS;
    } catch (const std::exception& e) {
        return CONV_FAILURE;
    }
}

// Build Weight Gradient Graph
ConvError_t build_wgrad_graph(
    cudnnHandle_t handle,
    ConvGraph_t* graph_out,
    const ConvTensorDescriptor_t* x_desc,
    const ConvTensorDescriptor_t* dy_desc,
    const ConvTensorDescriptor_t* dw_desc,
    const ConvConvolutionDescriptor_t* conv_desc,
    ConvDataType_t data_type)
{
    if (graph_out == nullptr || x_desc == nullptr || dy_desc == nullptr ||
        dw_desc == nullptr || conv_desc == nullptr) {
        return CONV_INVALID_VALUE;
    }
    try {
        using namespace cudnn_frontend;
        ConvGraph* conv_graph = new ConvGraph();

        auto graph = std::make_shared<graph::Graph>();
        auto cudnn_data_type = getCudnnDataType(data_type);
        graph->set_io_data_type(cudnn_data_type)
            .set_intermediate_data_type(cudnn_data_type)
            .set_compute_data_type(cudnn_data_type);

        // Create input tensor X
        std::vector<int64_t> x_dims(x_desc->dims, x_desc->dims + x_desc->num_dims);
        std::vector<int64_t> x_strides(x_desc->strides, x_desc->strides + x_desc->num_dims);
        auto X = graph->tensor(graph::Tensor_attributes()
                                   .set_name("input")
                                   .set_dim(x_dims)
                                   .set_stride(x_strides));

        // Create input tensor DY
        std::vector<int64_t> dy_dims(dy_desc->dims, dy_desc->dims + dy_desc->num_dims);
        std::vector<int64_t> dy_strides(dy_desc->strides, dy_desc->strides + dy_desc->num_dims);
        auto DY = graph->tensor(graph::Tensor_attributes()
                                    .set_name("grad_output")
                                    .set_dim(dy_dims)
                                    .set_stride(dy_strides));

        // Set convolution options
        std::vector<int64_t> padding(conv_desc->padding, conv_desc->padding + conv_desc->num_dims);
        std::vector<int64_t> stride(conv_desc->stride, conv_desc->stride + conv_desc->num_dims);
        std::vector<int64_t> dilation(conv_desc->dilation, conv_desc->dilation + conv_desc->num_dims);

        auto wgrad_options = graph::Conv_wgrad_attributes()
                                 .set_padding(padding)
                                 .set_stride(stride)
                                 .set_dilation(dilation);

        auto DW = graph->conv_wgrad(DY, X, wgrad_options);

        // Set output tensor DW properties
        std::vector<int64_t> dw_dims(dw_desc->dims, dw_desc->dims + dw_desc->num_dims);
        DW->set_dim(dw_dims).set_output(true);

        // Validate and build the graph
        auto status = graph->validate();
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->build_operation_graph(handle);
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->create_execution_plans({HeurMode_t::A});
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->check_support(handle);
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        status = graph->build_plans(handle);
        if (!status.is_good()) {
            delete conv_graph;
            return CONV_FAILURE;
        }

        conv_graph->graph_ptr = graph;
        conv_graph->input_tensors.push_back(X);
        conv_graph->input_tensors.push_back(DY);
        conv_graph->output_tensors.push_back(DW);

        *graph_out = static_cast<ConvGraph_t>(conv_graph);
        return CONV_SUCCESS;
    } catch (const std::exception& e) {
        return CONV_FAILURE;
    }
}

// Get Workspace Size
ConvError_t get_workspace_size(ConvGraph_t graph, size_t* workspace_size) {
    if (graph == nullptr || workspace_size == nullptr) {
        return CONV_INVALID_VALUE;
    }
    ConvGraph* conv_graph = static_cast<ConvGraph*>(graph);
    try {
        int64_t ws_size;
        auto status = conv_graph->graph_ptr->get_workspace_size(ws_size);
        if (status.is_good()) {
            *workspace_size = static_cast<size_t>(ws_size);
            return CONV_SUCCESS;
        } else {
            return CONV_FAILURE;
        }
    } catch (const std::exception& e) {
        return CONV_FAILURE;
    }
}

// Execute Graph
ConvError_t execute_graph(
    cudnnHandle_t handle,
    ConvGraph_t graph,
    void* input_ptrs[],
    void* output_ptrs[],
    void* workspace)
{
    if (graph == nullptr || input_ptrs == nullptr || output_ptrs == nullptr) {
        return CONV_INVALID_VALUE;
    }
    ConvGraph* conv_graph = static_cast<ConvGraph*>(graph);
    if (conv_graph->input_tensors.size() == 0 || conv_graph->output_tensors.size() == 0) {
        return CONV_INVALID_VALUE;
    }
    try {
        std::unordered_map<int64_t, void*> variant_pack;
        for (size_t i = 0; i < conv_graph->input_tensors.size(); ++i) {
            if (input_ptrs[i] == nullptr) {
                return CONV_INVALID_VALUE;
            }
            variant_pack[conv_graph->input_tensors[i]->get_uid()] = input_ptrs[i];
        }
        for (size_t i = 0; i < conv_graph->output_tensors.size(); ++i) {
            if (output_ptrs[i] == nullptr) {
                return CONV_INVALID_VALUE;
            }
            variant_pack[conv_graph->output_tensors[i]->get_uid()] = output_ptrs[i];
        }
        auto status = conv_graph->graph_ptr->execute(handle, variant_pack, workspace);
        if (status.is_good()) {
            return CONV_SUCCESS;
        } else {
            return CONV_FAILURE;
        }
    } catch (const std::exception& e) {
        return CONV_FAILURE;
    }
}

// Destroy Graph
void destroy_graph(ConvGraph_t graph) {
    if (graph != nullptr) {
        ConvGraph* conv_graph = static_cast<ConvGraph*>(graph);
        delete conv_graph;
    }
}

// Get Number of Inputs
ConvError_t get_num_inputs(ConvGraph_t graph, size_t* num_inputs) {
    if (graph == nullptr || num_inputs == nullptr) {
        return CONV_INVALID_VALUE;
    }
    ConvGraph* conv_graph = static_cast<ConvGraph*>(graph);
    *num_inputs = conv_graph->input_tensors.size();
    return CONV_SUCCESS;
}

// Get Number of Outputs
ConvError_t get_num_outputs(ConvGraph_t graph, size_t* num_outputs) {
    if (graph == nullptr || num_outputs == nullptr) {
        return CONV_INVALID_VALUE;
    }
    ConvGraph* conv_graph = static_cast<ConvGraph*>(graph);
    *num_outputs = conv_graph->output_tensors.size();
    return CONV_SUCCESS;
}

