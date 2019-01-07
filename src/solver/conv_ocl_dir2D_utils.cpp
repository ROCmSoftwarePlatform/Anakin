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

#include "miopen/solver.hpp"

namespace miopen {
namespace solver {

void addPoolingKernel(const ConvolutionContext& params, ConvSolution& result) {
    KernelInfo kernel2;
    int pooling_type = 0;
    int average_include = 0;

    kernel2.l_wk           = {256, 1, 1};
    kernel2.g_wk           = {64 * 64 * 40, 1, 1};
    kernel2.kernel_file    = "PoolingGen.cl";
    kernel2.kernel_name    = "mloPooling";
    kernel2.isMIOpenKernel = false;

    int bot_batch_stride   = params.poolingContext.in_width * params.poolingContext.in_height * params.poolingContext.n_inputs;
    int bot_channel_stride = params.poolingContext.in_width * params.poolingContext.in_height;

    int top_batch_stride   = params.poolingContext.out_width * params.poolingContext.out_height * params.poolingContext.n_outputs;
    int top_channel_stride = params.poolingContext.out_width * params.poolingContext.out_height;

    // set comp_options...
    kernel2.comp_options =
        std::string(" -DMLO_POOLING_OP_ID=") + std::to_string(params.poolingContext.pooling_type)
        + std::string(" -DMLO_POOLING_KERNEL_SZ0=") + std::to_string(params.poolingContext.kernel_size0)
        + std::string(" -DMLO_POOLING_KERNEL_SZ1=") + std::to_string(params.poolingContext.kernel_size0)
        + std::string(" -DMLO_POOLING_PAD0=") + std::to_string(params.poolingContext.pad0)
        + std::string(" -DMLO_POOLING_PAD1=") + std::to_string(params.poolingContext.pad1)
        + std::string(" -DMLO_POOLING_STRIDE0=") + std::to_string(params.poolingContext.kernel_stride0)
        + std::string(" -DMLO_POOLING_STRIDE1=") + std::to_string(params.poolingContext.kernel_stride1)
        + std::string(" -DMLO_POOLING_N_OUTPUTS=") + std::to_string(params.poolingContext.n_outputs)
        + std::string(" -DMLO_POOLING_N_CHANNELS=") + std::to_string(params.poolingContext.n_inputs)
        + std::string(" -DMLO_POOLING_GROUP_SZ0=8")
        + std::string(" -DMLO_POOLING_GROUP_SZ1=8")
        + std::string(" -DMLO_POOLING_BOT_BATCH_STRIDE=") + std::to_string(bot_batch_stride)
        + std::string(" -DMLO_POOLING_BOT_CHANNEL_STRIDE=") + std::to_string(bot_channel_stride)
        + std::string(" -DMLO_POOLING_BOT_STRIDE=") + std::to_string(params.poolingContext.in_width)
        + std::string(" -DMLO_POOLING_TOP_BATCH_STRIDE=") + std::to_string(top_batch_stride)
        + std::string(" -DMLO_POOLING_TOP_CHANNEL_STRIDE=") + std::to_string(top_channel_stride)
        + std::string(" -DMLO_POOLING_TOP_STRIDE=") + std::to_string(params.poolingContext.out_width)
        + std::string(" -DMLO_POOLING_BOT_WIDTH=") + std::to_string(params.poolingContext.in_width)
        + std::string(" -DMLO_POOLING_BOT_HEIGHT=") + std::to_string(params.poolingContext.in_height)
        + std::string(" -DMLO_POOLING_TOP_WIDTH=") + std::to_string(params.poolingContext.out_width)
        + std::string(" -DMLO_POOLING_TOP_HEIGHT=") + std::to_string(params.poolingContext.out_height)
        + std::string(" -DBATCH_NUM=") + std::to_string(params.poolingContext.batch_sz)
        + std::string(" -DAVERAGE_INCLUDE=") + std::to_string(average_include)
        + std::string(" -DCU_NUM=64")
        + std::string(" -DMLO_CONV_BIAS=0")
        + std::string(" -DMLO_CONV_PRELU=0")
        + std::string(" -DMIOPEN_USE_FP32=1")
        + std::string(" -DMIOPEN_USE_FP16=0");

    result.construction_params.push_back(kernel2);
}

} // namespace solver
} // namespace miopen
