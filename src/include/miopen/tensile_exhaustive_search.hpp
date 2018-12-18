/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_TENSILE_EXHAUSTIVE_SEARCH_HPP
#define GUARD_MIOPEN_TENSILE_EXHAUSTIVE_SEARCH_HPP

#include <miopen/config.h>
#include <miopen/serializable.hpp>
#include <iostream>
#include <string>

namespace miopen {
namespace solver {

struct TensilePerformanceConfig : Serializable<TensilePerformanceConfig> {
    std::string kernel_file = "";
    int local_worker_x       = 0;
    int global_worker_x      = 0;
    int local_worker_y       = 0;
    int global_worker_y      = 0;
    int local_worker_z       = 0;
    int global_worker_z      = 0;
    int param_size_1         = 0;
    int param_size_2         = 0;
    int param_size_3         = 0;
    double min_proc_time     = std::numeric_limits<float>::max();

    template <class Solution>
    void CopyTo(Solution& iud) const {
        iud.kernel_file      = kernel_file;
        iud.local_worker_x   = local_worker_x;
        iud.local_worker_y   = local_worker_y;
        iud.local_worker_z   = local_worker_z;
        iud.global_worker_x  = global_worker_x;
        iud.global_worker_y  = global_worker_y;
        iud.global_worker_z  = global_worker_z;
        iud.param_size_1     = param_size_1;
        iud.param_size_2     = param_size_2;
        iud.param_size_3     = param_size_3;
        iud.min_proc_time    = min_proc_time;
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f) {
        f(self.kernel_file, "temp.kernel_file");
        f(self.local_worker_x, "temp.local_worker_x");
        f(self.local_worker_y, "temp.local_worker_y");
        f(self.local_worker_z, "temp.local_worker_z");
        f(self.global_worker_x, "temp.global_worker_x");
        f(self.global_worker_y, "temp.global_worker_y");
        f(self.global_worker_z, "temp.global_worker_z");
        f(self.param_size_1, "temp.param_size_1");
        f(self.param_size_2, "temp.param_size_2");
        f(self.param_size_3, "temp.param_size_3");
        f(self.min_proc_time, "temp.min_proc_time");
    }
};
} // namespace solver
} // namespace miopen

#endif
