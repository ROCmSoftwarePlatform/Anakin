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

#include "miopen/solver.hpp"

namespace miopen {
namespace solver {

void addPoolingKernel(const ConvolutionContext& params, ConvSolution& result)
{
    int _grp_tile0 = 8;
    int _grp_tile1 = 8;

    int _out_pix_tile0 = std::max(1, 8 / params.poolingContext.kernel_stride0);
    int _out_pix_tile1 = std::max(1, 8 / params.poolingContext.kernel_stride1);

    while(_out_pix_tile0 * _grp_tile0 > params.poolingContext.out_width * 2 && _out_pix_tile0 > 1)
    {
        _out_pix_tile0 >>= 1;
    }

    while(_out_pix_tile1 * _grp_tile1 > params.poolingContext.out_height * 2 && _out_pix_tile1 > 1)
    {
        _out_pix_tile1 >>= 1;
    }

    int g_wk_width =
        ((params.poolingContext.out_width + _grp_tile0 * _out_pix_tile0 - 1) / (_grp_tile0 * _out_pix_tile0));
    int g_wk_height =
        ((params.poolingContext.out_height + _grp_tile1 * _out_pix_tile1 - 1) / (_grp_tile1 * _out_pix_tile1));

    KernelInfo kernel2;
    if(params.poolingContext.kernel_size1 == 2 && params.poolingContext.kernel_size0 == 2)
    {
        kernel2.l_wk        = {256, 1, 1};
        kernel2.g_wk        = {64 * 64 * 40, 1, 1};
        kernel2.kernel_file = "BiasReLuPooling.cl";
        kernel2.kernel_name = "mloPooling";
        kernel2.isMIOpenKernel = false;
    }
    else
    {
        kernel2.l_wk        = {_grp_tile0, _grp_tile1, 1};
        kernel2.g_wk        = {g_wk_width * _grp_tile0,
                               g_wk_height * _grp_tile1,
                               params.poolingContext.n_inputs * params.poolingContext.batch_sz};
        kernel2.kernel_file = "MIOpenPooling.cl";
        kernel2.kernel_name = "mloPoolingG";
        kernel2.isMIOpenKernel = true;
    }

    kernel2.comp_options =
        std::string(" -DMLO_POOLING_OP_ID=") + std::to_string(params.poolingContext.pooling_type) +
        std::string(" -DMLO_POOLING_KERNEL_SZ0=") + std::to_string(params.poolingContext.kernel_size0) +
        std::string(" -DMLO_POOLING_KERNEL_SZ1=") + std::to_string(params.poolingContext.kernel_size1) +
        std::string(" -DMLO_POOLING_PAD0=") + std::to_string(params.poolingContext.pad0) +
        std::string(" -DMLO_POOLING_PAD1=") + std::to_string(params.poolingContext.pad1) +
        std::string(" -DMLO_POOLING_STRIDE0=") + std::to_string(params.poolingContext.kernel_stride0) +
        std::string(" -DMLO_POOLING_STRIDE1=") + std::to_string(params.poolingContext.kernel_stride1) +
        std::string(" -DMLO_POOLING_N_OUTPUTS=") + std::to_string(params.poolingContext.n_outputs) +
        std::string(" -DMLO_POOLING_N_CHANNELS=") + std::to_string(params.poolingContext.n_inputs) +
        std::string(" -DMLO_POOLING_N_HORIZ_OUT_PIX=") + std::to_string(_out_pix_tile0) +
        std::string(" -DMLO_POOLING_N_VERT_OUT_PIX=") + std::to_string(_out_pix_tile1) +
        std::string(" -DMLO_POOLING_GROUP_SZ0=") + std::to_string(_grp_tile0) +
        std::string(" -DMLO_POOLING_GROUP_SZ1=") + std::to_string(_grp_tile1) +
        std::string(" -DMLO_POOLING_BOT_WIDTH=") + std::to_string(params.poolingContext.in_width) +
        std::string(" -DMLO_POOLING_BOT_HEIGHT=") + std::to_string(params.poolingContext.in_height) +
        std::string(" -DMLO_POOLING_BOT_STRIDE=") + std::to_string(params.poolingContext.in_width) +
        std::string(" -DMLO_POOLING_BOT_CHANNEL_STRIDE=") + std::to_string(params.poolingContext.in_width * params.poolingContext.in_height) +
        std::string(" -DMLO_POOLING_BOT_BATCH_STRIDE=") +
            std::to_string(params.poolingContext.in_width * params.poolingContext.in_height * params.poolingContext.n_inputs) +
        std::string(" -DMLO_POOLING_TOP_WIDTH=") + std::to_string(params.poolingContext.out_width) +
        std::string(" -DMLO_POOLING_TOP_HEIGHT=") + std::to_string(params.poolingContext.out_height) +
        std::string(" -DMLO_POOLING_TOP_STRIDE=") + std::to_string(params.poolingContext.out_width) +
        std::string(" -DMLO_POOLING_TOP_CHANNEL_STRIDE=") + std::to_string(params.poolingContext.out_width * params.poolingContext.out_height) +
        std::string(" -DMLO_POOLING_TOP_BATCH_STRIDE=") +
            std::to_string(params.poolingContext.out_width * params.poolingContext.out_height * params.poolingContext.n_outputs) +
        std::string(" -DBATCH_NUM=") + std::to_string(params.poolingContext.batch_sz) +
        std::string(" -DCU_NUM=64") +
        std::string(" -DMLO_CONV_BIAS=0") +
        std::string(" -DMIOPEN_USE_FP32=1");
    result.construction_params.push_back(kernel2);
}

} // namespace solver
} // namespace miopen
