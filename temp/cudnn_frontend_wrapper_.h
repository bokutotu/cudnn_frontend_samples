#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef enum {
    SUCCESS = 0,
    FAILURE = 1,
    INVALID_VALUE = 2,
    NOT_SUPPORTED = 3,
    // Add more error codes as needed
} CudnnFrontendError_t;

typedef enum {
    DATA_TYPE_HALF,
    DATA_TYPE_FLOAT,
    DATA_TYPE_DOUBLE,
    // Add more as needed
} CudnnFrontendDataType_t;

typedef struct {
    size_t num_dims;
    int64_t dims[8];    // Maximum of 8 dimensions
    int64_t strides[8]; // Corresponding strides
} CudnnTensorDescriptor_t;

#include "conv.h"
#include "batch_norm.h"

#ifdef __cplusplus
}
#endif
