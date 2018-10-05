/*******************************************************************************
 *solver_conv_common.hpp
 *Created on: Oct. 5, 2018
 *******************************************************************************/

#ifndef SOLVER_CONV_COMMON_H
#define SOLVER_CONV_COMMON_H

#include <iostream>
#include <vector>
#define ALOGD(X) std::cout << X << std::endl;
#define ALOGE(X) std::cerr << X << std::endl;

namespace miopen {

enum DeviceType { GFX803, GFX900 };

struct Conv11Param {
    int global_split;
    int stride_per_iter;
    int tile_col;
    int tile_row;
    int wi_per_tile_col;
    int wi_per_tile_row;
    int code_branch;
    int code_method;
};

struct Conv1x1Type {
    int dev;
    int batch;
    int stride;
    int channel;
    int width;
    int output_num;
    char* kernel_name;
    Conv11Param params;
    bool isValid;
};

class ConvCommon {
public:
    ConvCommon() {}
    ~ConvCommon() {}
    void init();
    Conv1x1Type getKernelInfo(int dev, int stride, int channel, int width, int output_num);

private:
    std::vector<Conv1x1Type> conv1x1type;
};

} // namespace miopen

#endif
