#include "../include/cudnn_frontend_wrapper.h"

#include "batchnorm.h"

CudnnFrontendError_t create_batch_norm_descriptor(BatchNormDescriptor** desc, 
                                                  CudnnFrontendDataType_t data_type, 
                                                  const CudnnTensorShapeStride* shape,
                                                  float epsilon,
                                                  float momentum,
                                                  bool is_training) {

    *desc = new BatchNormDescriptor(*shape, is_training, data_type, epsilon, momentum);
    return CudnnFrontendError_t::SUCCESS;
}

CudnnFrontendError_t get_workspace_size(BatchNormDescriptor* desc, int64_t* workspace_size) {
    return desc->get_workspace_size(workspace_size);
}

CudnnFrontendError_t check_graph(BatchNormDescriptor* desc, cudnnHandle_t* handle) {
    return desc->check_graph(handle);
}

void batch_norm_desc_debug(BatchNormDescriptor* desc) {
    desc->debug_print();
}

CudnnFrontendError_t execute_batch_norm_forward_training(BatchNormDescriptor* desc, 
                                                         BatchNormExecutionBuffers* buffers,
                                                         void* workspace,
                                                         cudnnHandle_t* handle) {
    return desc->execute(handle, buffers, workspace);
}

CudnnFrontendError_t create_batch_norm_bckward_data_descriptor(BatchNormBkwdDescriptor** desc,
                                                               CudnnFrontendDataType_t data_type,
                                                               const CudnnTensorShapeStride* shape) {
    *desc = new BatchNormBkwdDescriptor(*shape, data_type);
    return CudnnFrontendError_t::SUCCESS;
}

CudnnFrontendError_t get_backward_data_workspace_size(BatchNormBkwdDescriptor* desc, int64_t* workspace_size) {
    return desc->get_workspace_size(workspace_size);
}

CudnnFrontendError_t check_backward_data_graph(BatchNormBkwdDescriptor* desc, cudnnHandle_t* handle) {
    return desc->check_graph(handle);
}

CudnnFrontendError_t execute_batch_norm_backward_data(BatchNormBkwdDescriptor* desc,
                                                      BatchNormBkwdExecutionBuffers* buffers,
                                                      void* workspace,
                                                      cudnnHandle_t* handle) {
    return desc->execute(handle, buffers, workspace);
}
