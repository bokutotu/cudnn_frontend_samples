#pragma once

#include "../include/cudnn_frontend_wrapper.h"
#include <cudnn_frontend.h>

struct BatchNormDescriptor {
    cudnn_frontend::graph::Graph graph;
};
