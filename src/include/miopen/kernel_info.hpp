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
/*
 * kernel_info.hpp
 *
 *  Created on: Sep 10, 2018
 *      Author: junpeng
 */

#ifndef SABER_FUNCS_IMPL_AMD_KERNEL_INFO_HPP_
#define SABER_FUNCS_IMPL_AMD_KERNEL_INFO_HPP_

#include <string>
#include <vector>

namespace miopen {
namespace solver {

struct KernelInfo {
    std::string comp_options;
    std::vector<size_t> l_wk;
    std::vector<size_t> g_wk;
    std::string kernel_file;
    std::string kernel_name;
    friend std::ostream& operator<<(std::ostream& os, const KernelInfo& k);
    bool isMIOpenKernel = true;
    int tensile_slot_size;
    int tensile_l2_size;
    int tensile_dbg_size;
};

} // namespace solver
} // namespace miopen

#endif /* SABER_FUNCS_IMPL_AMD_KERNEL_INFO_HPP_ */
