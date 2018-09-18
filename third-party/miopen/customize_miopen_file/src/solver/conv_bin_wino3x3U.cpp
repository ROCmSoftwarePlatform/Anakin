/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include "miopen/env.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_3X3)

namespace miopen {
namespace solver {

bool ConvBinWinograd3x3U::IsApplicable(const ConvolutionContext& params) const
{
    if(!params.use_binaries)
    {
        return false;
    }

    if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_3X3{}))
    {
        return false;
    }

    // Check if device is able to run this kernel.
    const auto name = params.GetStream().GetDeviceName();
    // clang-format off
    if (! ((name == "gfx803" && (params.rmv == rocm_meta_version::V1
                              || params.rmv == rocm_meta_version::V2
                              || params.rmv == rocm_meta_version::V3
                              || params.rmv == rocm_meta_version::AMDHSA_1_0 ))
        || (name == "gfx900" && (params.rmv == rocm_meta_version::V3
                              || params.rmv == rocm_meta_version::AMDHSA_1_0))
        || (name == "gfx906" && params.rmv == rocm_meta_version::AMDHSA_1_0)))
        return false;
    // clang-format on

    // Check if kernel is suitable for the problem description
    // and able to correctly run with given parameters.
    const auto device_is_gfx8         = (name.find("gfx8") != std::string::npos);
    const auto grid_workgroup_count_x = params.GetStream().GetMaxComputeUnits();
    assert(params.weights_layout.length() == 0); // weights_layout is not supported yet.
    // clang-format off
    return params.pad0 == 1
        && params.pad1 == 1
        && params.kernel_size0 == 3
        && params.kernel_size1 == 3
        && params.kernel_stride0 == 1
        && params.kernel_stride1 == 1
        && params.batch_sz < std::pow(2, 16)
        && params.n_inputs < std::pow(2, 16)
        && params.n_outputs < std::pow(2, 16)
        && params.in_height < std::pow(2, 16)
        && params.in_width < std::pow(2, 16)
        && grid_workgroup_count_x < std::pow(2, 16)
        && (params.n_inputs * params.in_height * params.in_width) <= std::pow(2, 28)
        && (params.n_outputs * params.in_height * params.in_width) <= std::pow(2, 28)
        && (params.n_inputs * params.kernel_size0 * params.kernel_size1) <= std::pow(2, 28)
        && (params.n_outputs * params.kernel_size0 * params.kernel_size1) <= std::pow(2, 28)
        && params.n_inputs % 2 == 0 && params.n_inputs >= (device_is_gfx8 ? 16 : 18)
        && params.float_size == 32
        && params.in_layout == "NCHW";
        /// \todo _n_inputs > 18 is a requirement of the v7 shader and NOT a dependency on gfx9
        /// The current way of implemenation is a hack as gfx8 uses v3.0 shader and gfx9 uses v7.
        /// && (isForwardDirection() ? _weights_layout == "KCHW" : _weights_layout == "CKHW" )
        /// Actually, K<->C flpping is controlled by separate flag, so we can support either
        /// layout in both directions.

    // clang-format on
}

ConvSolution ConvBinWinograd3x3U::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
    const auto n_groups = params.GetStream().GetMaxComputeUnits();
    const auto name     = params.GetStream().GetDeviceName();

    KernelInfo kernel;
    KernelInfo kernel2;

    kernel.g_wk.clear();
    kernel.g_wk.push_back(512 * n_groups);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.clear();
    kernel.l_wk.push_back(512);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.kernel_name = "sp3AsmConv3x3F";
    if(name == "gfx803")
    {
        if(params.rmv == rocm_meta_version::V1)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m10.so";
        else if(params.rmv == rocm_meta_version::V2)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m21.so";
        else if(params.rmv == rocm_meta_version::V3)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m30.so";
        else if(params.rmv == rocm_meta_version::AMDHSA_1_0)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_md10.so";
        else
            MIOPEN_THROW("conv_3x3_wheel_alpha_v3_0b_gfx803: Unsupported metadata version.");
    }
    else if(name == "gfx900")
    {
        if(params.rmv == rocm_meta_version::V3)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900.so";
        else if(params.rmv == rocm_meta_version::AMDHSA_1_0)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900_md10.so";
        else
            MIOPEN_THROW("conv_3x3_wheel_alpha_v7_0_3b_gfx900: Unsupported metadata version.");
    }
    else if(name == "gfx906")
    {
        if(params.rmv == rocm_meta_version::AMDHSA_1_0)
            kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx906_md10.so";
        else
            MIOPEN_THROW("conv_3x3_wheel_alpha_v7_0_3b_gfx906: Unsupported metadata version.");
    }
    else
    {
        MIOPEN_THROW("conv_3x3_wheel_alpha_v7_0_3b: Unsupported device.");
    }
    result.construction_params.push_back(kernel);

    // Start to do pooling...
    if (params.has_pooling) {
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
            std::string(" -DMLO_POOLING_N_HORIZ_OUT_PIX=") +
            std::to_string(_out_pix_tile0) +
            std::string(" -DMLO_POOLING_N_VERT_OUT_PIX=") +
            std::to_string(_out_pix_tile1) +
            std::string(" -DMLO_POOLING_GROUP_SZ0=") + std::to_string(_grp_tile0) +
            std::string(" -DMLO_POOLING_GROUP_SZ1=") + std::to_string(_grp_tile1) +
            std::string(" -DMLO_POOLING_BOT_WIDTH=") + std::to_string(params.poolingContext.in_width) +
            std::string(" -DMLO_POOLING_BOT_HEIGHT=") + std::to_string(params.poolingContext.in_height) +
            std::string(" -DMLO_POOLING_BOT_STRIDE=") + std::to_string(params.poolingContext.in_width) +
            std::string(" -DMLO_POOLING_BOT_CHANNEL_STRIDE=") +
            std::to_string(params.poolingContext.in_width * params.poolingContext.in_height) +
            std::string(" -DMLO_POOLING_BOT_BATCH_STRIDE=") +
            std::to_string(params.poolingContext.in_width * params.poolingContext.in_height * params.poolingContext.n_inputs) +

            std::string(" -DMLO_POOLING_TOP_WIDTH=") + std::to_string(params.poolingContext.out_width) +
            std::string(" -DMLO_POOLING_TOP_HEIGHT=") + std::to_string(params.poolingContext.out_height) +
            std::string(" -DMLO_POOLING_TOP_STRIDE=") + std::to_string(params.poolingContext.out_width) +
            std::string(" -DMLO_POOLING_TOP_CHANNEL_STRIDE=") +
            std::to_string(params.poolingContext.out_width * params.poolingContext.out_height) +
            std::string(" -DMLO_POOLING_TOP_BATCH_STRIDE=") +
            std::to_string(params.poolingContext.out_width * params.poolingContext.out_height * params.poolingContext.n_outputs) +
            std::string(" -DBATCH_NUM=") + std::to_string(params.poolingContext.batch_sz) +
            std::string(" -DCU_NUM=64") + std::string(" -DMLO_CONV_BIAS=0") +
            std::string(" -DMIOPEN_USE_FP32=1");
        result.construction_params.push_back(kernel2);
    }

    return result;
}
} // namespace solver
} // namespace miopen
