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
#include "miopen/env.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_3X3)

namespace miopen {
namespace solver {

static bool isPoolAlign(const ConvolutionContext& params)
{
    int out_width  = params.out_width;
    int out_height = params.out_height;
    int kernel_size0 = params.poolingContext.kernel_size0;
    int kernel_size1 = params.poolingContext.kernel_size1;
    int kernel_stride0 = params.poolingContext.kernel_stride0;
    int kernel_stride1 = params.poolingContext.kernel_stride1;
    int pad0 = params.poolingContext.pad0;
    int pad1 = params.poolingContext.pad1;

    return ((out_width + 2 * pad0 - kernel_size0) % kernel_stride0 == 0)
           && ((out_height + 2 * pad1 - kernel_size1) % kernel_stride1 == 0);
}

bool ConvBinWinograd3x3U::IsApplicable(const ConvolutionContext& params) const {
    if (!params.use_binaries) {
        return false;
    }

    if (miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_3X3 {})) {
        return false;
    }

    // Check if device is able to run this kernel.
    const auto name = params.GetStream().GetDeviceName();

    // clang-format off
    if (!((name == "gfx803" && (params.rmv == rocm_meta_version::V1
                                || params.rmv == rocm_meta_version::V2
                                || params.rmv == rocm_meta_version::V3
                                || params.rmv == rocm_meta_version::AMDHSA_1_0))
            || (name == "gfx900" && (params.rmv == rocm_meta_version::V3
                                     || params.rmv == rocm_meta_version::AMDHSA_1_0))
            || (name == "gfx906" && params.rmv == rocm_meta_version::AMDHSA_1_0))) {
        return false;
    }

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
           && ((params.kernel_stride0 == 1 && params.kernel_stride1 == 1)
               || (params.kernel_stride0 == 2 && params.kernel_stride1 == 2))
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

ConvSolution ConvBinWinograd3x3U::GetSolution(const ConvolutionContext& params) const {
    ConvSolution result;
    const auto n_groups = params.GetStream().GetMaxComputeUnits();
    const auto name     = params.GetStream().GetDeviceName();

    result.status = miopenStatusInternalError;

    KernelInfo kernel;
    //KernelInfo kernel2;

    kernel.g_wk.clear();
    kernel.g_wk.push_back(32768);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.l_wk.clear();
    kernel.l_wk.push_back(512);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.kernel_name = "sp3AsmConv3x3F";

    if (name == "gfx803") {
        if (params.rmv == rocm_meta_version::V1) {
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m10.so";
        } else if (params.rmv == rocm_meta_version::V2) {
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m21.so";
        } else if (params.rmv == rocm_meta_version::V3) {
            kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_m30.so";
        } else if (params.rmv == rocm_meta_version::AMDHSA_1_0) {
            if (params.has_active && params.bias) {
                if (params.kernel_stride0 == 1) {
                    if (params.has_pooling
                        && isPoolAlign(params)
                        && params.kernel_stride0 == 1
                        && params.poolingContext.pooling_type == 1 //Pooling_max
                        && params.poolingContext.kernel_size0 == 2
                        && params.poolingContext.kernel_size1 == 2
                        && params.poolingContext.kernel_stride0 == 2
                        && params.poolingContext.kernel_stride1 == 2
                        && params.poolingContext.pad0 == 0
                        && params.poolingContext.pad1 == 0) {
                        kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_md10_bias_prelu_pooling.so";
                    } else if ((params.n_inputs == 128 && params.n_outputs == 128
                                && params.in_height == 28 && params.in_width == 28 && params.batch_sz == 1)
                               || (params.n_inputs == 1024 && params.n_outputs == 1024
                                && params.in_height == 7 && params.in_width == 7 && params.batch_sz <= 2)
                               || (params.n_inputs == 512 && params.n_outputs == 512
                                && params.in_height == 7 && params.in_width == 7 && params.batch_sz <= 4)
                               || (params.n_inputs == 512 && params.n_outputs == 512
                                && params.in_height == 14 && params.in_width == 14 && params.batch_sz == 1)
                               || (params.n_inputs == 256 && params.n_outputs == 256
                                && params.in_height == 14 && params.in_width == 14 && params.batch_sz <= 2)
                               || (params.n_outputs == 384 && params.in_height == 13
                                && params.in_width == 13 && params.batch_sz <= 1)
                               || (params.n_outputs == 128 && params.in_height == 6
                                && params.in_width == 64 && params.batch_sz <= 2)
                               || (params.n_outputs == 64 && params.in_height == 12
                                && params.in_width == 128 && params.batch_sz <= 1)) {
                        //todo: remove n_inputs = n_outputs
                        kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_md10_bias_prelu_sw.so";
                    } else {
                        kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_md10_bias_prelu.so";
                    }
                } else if (params.kernel_stride0 == 2) {
                    kernel.kernel_file =   "conv_3x3_wheel_alpha_v3_0b_gfx803_md10_bias_prelu_stride2.so";
                } else {
                    return result;
                }
            } else if (params.has_active && !params.bias) {
                if (params.kernel_stride0 == 1) {
                    kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_md10_prelu.so";
                } else {
                    return result;
                }
            } else if (!params.has_active && params.bias) {
                if (params.kernel_stride0 == 1) {
                    kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_md10_bias.so";
                } else if (params.kernel_stride0 == 2) {
                    kernel.kernel_file =   "conv_3x3_wheel_alpha_v3_0b_gfx803_md10_bias_prelu_stride2.so"
                                           ;
                } else {
                    return result;
                }
            } else {
                if (params.kernel_stride0 == 1) {
                    kernel.kernel_file = "conv_3x3_wheel_alpha_v3_0b_gfx803_md10.so";
                } else {
                    return result;
                }
            }
        } else {
            MIOPEN_THROW("conv_3x3_wheel_alpha_v3_0b_gfx803: Unsupported metadata version.");
        }
    } else if (name == "gfx900") {
        if (params.rmv == rocm_meta_version::V3) {
            kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900.so";
        } else if (params.rmv == rocm_meta_version::AMDHSA_1_0) {
            if (params.has_active && params.bias) {
                if (params.kernel_stride0 == 1) {
                    if ((params.n_inputs == 128 && params.n_outputs == 128
                            && params.in_height == 28 && params.in_width == 28 && params.batch_sz == 1)
                            || (params.n_inputs == 1024 && params.n_outputs == 1024
                                && params.in_height == 7 && params.in_width == 7 && params.batch_sz <= 2)
                            || (params.n_inputs == 512 && params.n_outputs == 512
                                && params.in_height == 7 && params.in_width == 7 && params.batch_sz <= 4)
                            || (params.n_inputs == 512 && params.n_outputs == 512
                                && params.in_height == 14 && params.in_width == 14 && params.batch_sz == 1)
                            || (params.n_inputs == 256 && params.n_outputs == 256
                                && params.in_height == 14 && params.in_width == 14 && params.batch_sz <= 2)) {
                        //todo: remove n_inputs = n_outputs
                        kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900_md10_bias_prelu_sw.so";
                    } else {
                        kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900_md10_bias_prelu.so";
                    }
                } else if (params.kernel_stride0 == 2
                           && params.n_inputs == 1024 && params.in_height == 14 && params.in_width == 14) {
                    kernel.kernel_file =   "conv_3x3_wheel_alpha_v7_0_3b_gfx900_md10_bias_prelu_stride2.so"
                                           ;
                } else {
                    return result;
                }

                if (params.has_pooling && params.kernel_stride0 == 1
                        && params.poolingContext.pooling_type == 1 //Pooling_max
                        && params.poolingContext.kernel_size0 == 2
                        && params.poolingContext.kernel_size1 == 2
                        && params.poolingContext.kernel_stride0 == 2
                        && params.poolingContext.kernel_stride1 == 2
                        && params.poolingContext.pad0 == 0
                        && params.poolingContext.pad1 == 0) {
                    kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900_md10_bias_prelu_pooling.so";
                }
            } else if (params.has_active && !params.bias) {
                if (params.kernel_stride0 == 1) {
                    kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900_md10_prelu.so";
                } else {
                    return result;
                }
            } else if (!params.has_active && params.bias) {
                if (params.kernel_stride0 == 1) {
                    kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900_md10_bias.so";
                } else {
                    return result;
                }
            } else {
                if (params.kernel_stride0 == 1) {
                    kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx900_md10.so";
                } else {
                    return result;
                }
            }
        } else {
            MIOPEN_THROW("conv_3x3_wheel_alpha_v7_0_3b_gfx900: Unsupported metadata version.");
        }
    } else if (name == "gfx906") {
        if (params.rmv == rocm_meta_version::AMDHSA_1_0) {
            kernel.kernel_file = "conv_3x3_wheel_alpha_v7_0_3b_gfx906_md10.so";
        } else {
            MIOPEN_THROW("conv_3x3_wheel_alpha_v7_0_3b_gfx906: Unsupported metadata version.");
        }
    } else {
        MIOPEN_THROW("conv_3x3_wheel_alpha_v7_0_3b: Unsupported device.");
    }

    result.status = miopenStatusSuccess;
    result.construction_params.push_back(kernel);

    // Start to do pooling...
    if (params.has_pooling
            && (kernel.kernel_file != "conv_3x3_wheel_alpha_v7_0_3b_gfx900_md10_bias_prelu_pooling.so")
            && (kernel.kernel_file != "conv_3x3_wheel_alpha_v3_0b_gfx803_md10_bias_prelu_pooling.so")) {
        addPoolingKernel(params, result);
    }

    return result;
}
} // namespace solver
} // namespace miopen