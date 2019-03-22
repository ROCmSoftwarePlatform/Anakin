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

    const PoolingContext& pool_param = params.poolingContext;

    KernelInfo kernel2;

    if (pool_param.kernel_size1 * pool_param.kernel_size0 >= 32
            && ((pool_param.kernel_size1 <= pool_param.kernel_stride1
                 && pool_param.kernel_size0 <= pool_param.kernel_stride0)
                || pool_param.out_height * pool_param.out_width == 1)) {
        int output_size  = pool_param.batch_sz * pool_param.n_outputs * pool_param.out_height *
                           pool_param.out_width;

        int group_size   = 256;
        int group_size_0 = 256;  // adder

        while (group_size_0 * 8 > pool_param.kernel_size1 * pool_param.kernel_size0 && group_size_0 > 1) {
            group_size_0 = group_size_0 >> 1;
        }

        int group_size_1 = group_size / group_size_0;

        int global_size_0 = group_size_0;
        int global_size_1 = (output_size + group_size_1 - 1) / group_size_1 * group_size_1;

        kernel2.l_wk         = {group_size_0, group_size_1, 1};
        kernel2.g_wk         = {global_size_0, global_size_1, 1};
        kernel2.kernel_file  = "PoolingGeneral.cl";
        kernel2.kernel_name  = "PoolingGeneral";
        kernel2.isMIOpenKernel = false;

        kernel2.comp_options = std::string(" -DGROUP_SIZE=") + std::to_string(group_size)
                               + std::string(" -DGROUP_SIZE_0=") + std::to_string(group_size_0)
                               + std::string(" -DGROUP_SIZE_1=") + std::to_string(group_size_1)
                               + std::string(" -DPOOLING_TYPE=") + std::to_string(pool_param.pooling_type)
                               + std::string(" -DADDER=") + std::to_string(group_size_0);
    } else {
        kernel2.l_wk           = {256, 1, 1};
        kernel2.g_wk           = {64 * 64 * 40, 1, 1};
        kernel2.kernel_file    = "PoolingGen.cl";
        kernel2.kernel_name    = "mloPooling";
        kernel2.isMIOpenKernel = false;

        int bot_batch_stride   = pool_param.in_width * pool_param.in_height * pool_param.n_inputs;
        int bot_channel_stride = pool_param.in_width * pool_param.in_height;

        int top_batch_stride   = pool_param.out_width * pool_param.out_height * pool_param.n_outputs;
        int top_channel_stride = pool_param.out_width * pool_param.out_height;

        // set comp_options...
        kernel2.comp_options =
            std::string(" -DMLO_POOLING_OP_ID=") + std::to_string(pool_param.pooling_type)
            + std::string(" -DMLO_POOLING_KERNEL_SZ0=") + std::to_string(pool_param.kernel_size0)
            + std::string(" -DMLO_POOLING_KERNEL_SZ1=") + std::to_string(pool_param.kernel_size1)
            + std::string(" -DMLO_POOLING_PAD0=") + std::to_string(pool_param.pad0)
            + std::string(" -DMLO_POOLING_PAD1=") + std::to_string(pool_param.pad1)
            + std::string(" -DMLO_POOLING_STRIDE0=") + std::to_string(pool_param.kernel_stride0)
            + std::string(" -DMLO_POOLING_STRIDE1=") + std::to_string(pool_param.kernel_stride1)
            + std::string(" -DMLO_POOLING_N_OUTPUTS=") + std::to_string(pool_param.n_outputs)
            + std::string(" -DMLO_POOLING_N_CHANNELS=") + std::to_string(pool_param.n_inputs)
            + std::string(" -DMLO_POOLING_GROUP_SZ0=8")
            + std::string(" -DMLO_POOLING_GROUP_SZ1=8")
            + std::string(" -DMLO_POOLING_BOT_BATCH_STRIDE=") + std::to_string(bot_batch_stride)
            + std::string(" -DMLO_POOLING_BOT_CHANNEL_STRIDE=") + std::to_string(bot_channel_stride)
            + std::string(" -DMLO_POOLING_BOT_STRIDE=") + std::to_string(pool_param.in_width)
            + std::string(" -DMLO_POOLING_TOP_BATCH_STRIDE=") + std::to_string(top_batch_stride)
            + std::string(" -DMLO_POOLING_TOP_CHANNEL_STRIDE=") + std::to_string(top_channel_stride)
            + std::string(" -DMLO_POOLING_TOP_STRIDE=") + std::to_string(pool_param.out_width)
            + std::string(" -DMLO_POOLING_BOT_WIDTH=") + std::to_string(pool_param.in_width)
            + std::string(" -DMLO_POOLING_BOT_HEIGHT=") + std::to_string(pool_param.in_height)
            + std::string(" -DMLO_POOLING_TOP_WIDTH=") + std::to_string(pool_param.out_width)
            + std::string(" -DMLO_POOLING_TOP_HEIGHT=") + std::to_string(pool_param.out_height)
            + std::string(" -DBATCH_NUM=") + std::to_string(pool_param.batch_sz)
            + std::string(" -DCU_NUM=64")
            + std::string(" -DMLO_CONV_BIAS=0")
            + std::string(" -DMLO_CONV_PRELU=0")
            + std::string(" -DMIOPEN_USE_FP32=1")
            + std::string(" -DMIOPEN_USE_FP16=0");
    }

    result.construction_params.push_back(kernel2);
}

} // namespace solver
} // namespace miopen
