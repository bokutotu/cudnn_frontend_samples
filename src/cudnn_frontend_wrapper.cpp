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

CudnnFrontendError_t execute_batch_norm_forward_training(BatchNormDescriptor* desc, 
                                                         BatchNormExecutionBuffers* buffers,
                                                         void* workspace,
                                                         cudnnHandle_t* handle) {
    return desc->execute(handle, buffers, workspace);
}
