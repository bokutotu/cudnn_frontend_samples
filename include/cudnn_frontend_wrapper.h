#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <cudnn.h>

typedef enum {
    SUCCESS = 0,
    FAILURE = 1,
    INVALID_VALUE = 2,
    NOT_SUPPORTED = 3,
    // Add more error codes as needed
} CudnnFrontendError_t;

typedef enum {
    DATA_TYPE_HALF = 0,
    DATA_TYPE_FLOAT = 1,
    DATA_TYPE_DOUBLE = 2,
    // Add more as needed
} CudnnFrontendDataType_t;

typedef struct {
    size_t num_dims;
    int64_t dims[8];
    int64_t strides[8];
} CudnnTensorShapeStride;

typedef struct BatchNormDescriptor BatchNormDescriptor;

typedef struct {
    void* X;
    void* mean;
    void* inv_variance;
    void* scale;
    void* bias;
    void* peer_stats_0;
    void* peer_stats_1;
    void* prev_running_mean;
    void* prev_running_var;
    void* next_running_mean;
    void* next_running_var;
    void* Y;
} BatchNormExecutionBuffers;

CudnnFrontendError_t create_batch_norm_descriptor(BatchNormDescriptor** desc, 
                                                  CudnnFrontendDataType_t data_type, 
                                                  const CudnnTensorShapeStride* shape,
                                                  float epsilon,
                                                  float momentum,
                                                  bool is_training);

void batch_norm_desc_debug(BatchNormDescriptor* desc);

CudnnFrontendError_t check_graph(BatchNormDescriptor* desc, cudnnHandle_t* handle);

CudnnFrontendError_t get_workspace_size(BatchNormDescriptor* desc, int64_t* workspace_size);

CudnnFrontendError_t execute_batch_norm_forward_training(BatchNormDescriptor* desc, 
                                                         BatchNormExecutionBuffers* buffers,
                                                         void* workspace,
                                                         cudnnHandle_t* handle);

typedef struct BatchNormBkwdDescriptor BatchNormBkwdDescriptor;

typedef struct {
    void* X;
    void* DY;
    void* scale;
    void* mean;
    void* inv_variance;
    void* dscale;
    void* dbias;
    void* DX;
    void* peer_stats_0;
    void* peer_stats_1;
} BatchNormBkwdExecutionBuffers;

CudnnFrontendError_t create_batch_norm_backward_data_descriptor(BatchNormBkwdDescriptor** desc, 
                                                                CudnnFrontendDataType_t data_type, 
                                                                const CudnnTensorShapeStride* shape);

CudnnFrontendError_t check_backward_data_graph(BatchNormBkwdDescriptor* desc, 
                                               cudnnHandle_t* handle);

CudnnFrontendError_t get_backward_data_workspace_size(BatchNormBkwdDescriptor* desc, 
                                                      int64_t* workspace_size);

// void batch_norm_backward_data_desc_debug(BatchNormBackwardDataDescriptor* desc);

CudnnFrontendError_t execute_batch_norm_backward_data(BatchNormBkwdDescriptor* desc,
                                                      BatchNormBkwdExecutionBuffers* buffers,
                                                      void* workspace,
                                                      cudnnHandle_t* handle);

typedef struct {
    void* X;
    void* filter;
    void* Y;
} ConvBufers;

// padding, stride, dilation infomation, this struct use for 2d and 1d.
// please name of this struct
typedef struct {
    int64_t padding[2];
    int64_t stride[2];
    int64_t dilation[2];
    int64_t num_dims;
} ConvInfo;

typedef struct ConvDescriptor ConvDescriptor;

CudnnFrontendError_t create_conv_descriptor(ConvDescriptor** desc, 
                                            CudnnFrontendDataType_t data_type, 
                                            const CudnnTensorShapeStride* x_shape,
                                            const CudnnTensorShapeStride* w_shape,
                                            const CudnnTensorShapeStride* y_shape,
                                            const ConvInfo* info);

CudnnFrontendError_t check_conv_graph(ConvDescriptor* desc, cudnnHandle_t* handle);

CudnnFrontendError_t get_conv_workspace_size(ConvDescriptor* desc, int64_t* workspace_size);

CudnnFrontendError_t execute_conv_forward(ConvDescriptor* desc, 
                                          ConvBufers* buffers,
                                          cudnnHandle_t* handle);


#ifdef __cplusplus
}
#endif
