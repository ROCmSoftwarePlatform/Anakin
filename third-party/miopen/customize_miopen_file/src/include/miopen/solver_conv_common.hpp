/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
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
    std::string kernel_name;
    Conv11Param params;
};

class ConvCommon {
public:
    ConvCommon() {}
    ~ConvCommon() {
        for (auto it : conv1x1type) {
            delete it;
        }
    }
    void init();
    Conv1x1Type*
    getKernelInfo(int dev, int batch, int stride, int channel, int width, int output_num);

private:
    std::vector<Conv1x1Type*> conv1x1type;
};

} // namespace miopen

#endif
