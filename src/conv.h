#ifdef __cplusplus
extern "C" {
#endif

#include <cudnn_frontend.h>

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

GraphComponentsHandle create_graph(
    Shape* input_shape,
    Shape* weight_shape,
    Shape* output_shape,
    ConvParams* conv_params);

bool execute_graph(
    cudnnHandle_t handle,
    GraphComponentsHandle components_handle,
    float* dy_tensor,
    float* w_tensor,
    float* dx_tensor);

void destroy_graph_components(GraphComponentsHandle components_handle);

#ifdef __cplusplus
}
#endif
