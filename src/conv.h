#ifdef __cplusplus
extern "C" {
#endif

#include <cudnn.h>

typedef void* GraphComponentsHandle;

typedef struct {
    long int* dim;
    long int* stride;
    int size;
} Shape;

typedef struct {
    long int* padding;
    long int* stride;
    long int* dilation;
    int size;
} ConvParams;

// Forward convolution
GraphComponentsHandle create_forward_graph(
    cudnnHandle_t handle,
    Shape* input_shape,
    Shape* weight_shape,
    Shape* output_shape,
    ConvParams* conv_params);

// Backward data convolution
GraphComponentsHandle create_backward_data_graph(
    cudnnHandle_t handle,
    Shape* input_shape,
    Shape* weight_shape,
    Shape* output_shape,
    ConvParams* conv_params);

// Backward filter convolution
GraphComponentsHandle create_backward_filter_graph(
    cudnnHandle_t handle,
    Shape* input_shape,
    Shape* weight_shape,
    Shape* output_shape,
    ConvParams* conv_params);

// Execute graph
bool execute_graph(
    cudnnHandle_t handle,
    GraphComponentsHandle components_handle,
    void* tensor_a,
    void* tensor_b,
    void* tensor_c);

// Destroy graph components
void destroy_graph_components(GraphComponentsHandle components_handle);

#ifdef __cplusplus
}
#endif
