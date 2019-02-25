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

#ifndef GUARD_MIOPEN_GEMM_UTILS_H_
#define GUARD_MIOPEN_GEMM_UTILS_H_

#include <iostream>
#include "miopen/handle.hpp"

namespace miopen {
namespace tempfix {
void set_offsets_to_uint(std::string& clstr, int times);
} // namespace tempfix
KernelInvoke transpose_NCHW2CNHW(Handle& handle,
                          int n,
                          int c,
                          int h_in,
                          int w_in,
                          int h_out,
                          int w_out,
                          int in_offset,
                          int out_offset,
                          int h_stride,
                          int w_stride);

KernelInvoke transpose_CNHW2NCHW(Handle& handle,
                          int n,
                          int c,
                          int h_out,
                          int w_out,
                          int h_in,
                          int w_in,
                          int in_offset,
                          int out_offset,
                          int h_stride,
                          int w_stride,
                          bool isBias);
} // namespace miopen
#endif
