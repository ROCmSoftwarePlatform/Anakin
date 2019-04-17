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

//#include "miopen/solver.hpp"
#include "miopen/solver_conv_common.hpp"

namespace miopen {
namespace solver {

struct Arguments {
    int tile_col;
    int tile_row;
};

#define ARG_SIZE    4

Arguments args[] = {
    {64, 32},
    {32, 32},
    {32, 16},
    {16, 16},
};

bool ConvOclDirectFwd1x1AMD::IsApplicable(const ConvolutionContext& params) const {
    bool result =
        params.direction.IsForward() && (params.kernel_size0 == 1 && params.kernel_size1 == 1)
        && /*(params.batch_sz <= 8) &&*/ (params.kernel_stride0 <= 2)
        && (params.kernel_stride0 == params.kernel_stride1)
        && (params.pad0 == 0 && params.pad1 == 0);
#if 0
    && (params.n_inputs == 16 || params.n_inputs == 24 || params.n_inputs == 32
        || params.n_inputs == 64 || params.n_inputs == 96 || params.n_inputs == 128
        || params.n_inputs == 144 || params.n_inputs == 160 || params.n_inputs == 192
        || params.n_inputs == 256 || params.n_inputs == 320 || params.n_inputs == 384
        || params.n_inputs == 512 || params.n_inputs == 576 || params.n_inputs == 960
        || params.n_inputs == 1024 || params.n_inputs == 1280 || params.n_inputs == 2048)
    && ((params.in_height == params.in_width);
        && (params.in_height == 1 || params.in_height == 7 || params.in_height == 14
            || params.in_height == 28 || params.in_height == 56 || params.in_height == 112))
    && (params.n_outputs == 16 || params.n_outputs == 24 || params.n_outputs == 32
        || params.n_outputs == 64 || params.n_outputs == 96 || params.n_outputs == 128
        || params.n_outputs == 144 || params.n_outputs == 160 || params.n_outputs == 192
        || params.n_outputs == 256 || params.n_outputs == 320 || params.n_outputs == 384
        || params.n_outputs == 512 || params.n_outputs == 576 || params.n_outputs == 960
        || params.n_outputs == 1000 || params.n_outputs == 1024 || params.n_outputs == 1280
        || params.n_outputs == 2048);

    if (result) {
        int dev;
        const auto dev_name = params.GetStream().GetDeviceName();

        if (dev_name == "gfx803") {
            dev = GFX803;
        } else if (dev_name == "gfx900") {
            dev = GFX900;
        }

        ConvCommon cc;
        Conv1x1Type conv11_param;
        result = cc.getKernelInfo(
                     dev,
                     params.batch_sz,
                     params.kernel_stride0,
                     params.n_inputs,
                     params.in_width,
                     params.n_outputs,
                     conv11_param);

        if (result) {
            ALOGD("ConvOclDirectFwd1x1AMD::IsApplicable result:" << result << " kernel name="
                  << conv11_param.kernel_name);
        }
    }

#endif

    return result;
}

ConvSolution ConvOclDirectFwd1x1AMD::GetSolution(
    const ConvolutionContext& params,
    const LegacyPerformanceConfig& searched_params) const {
    ConvSolution result;
    // searched_params.CopyTo(result);
    result.min_proc_time = searched_params.min_proc_time;

    ConvCommon cc;
    Conv1x1Type conv11_param;
    bool ret = cc.getKernelInfo(
                   params,
                   conv11_param);
    KernelInfo kernelInfo;

    if (ret) {
        kernelInfo.kernel_file = conv11_param.kernel_name;

        if (conv11_param.kernel_name == "Conv1x1.cl") {
            kernelInfo.isMIOpenKernel = true;


            if (conv11_param.kernel_method == 1) {
                kernelInfo.kernel_name = "conv1x1_act";

                if (cc._usemacro) {
                    kernelInfo.comp_options =
                        std::string(" -DMACRO")
                        + std::string(" -DBIAS=") + std::to_string(params.bias) + std::string(" -DN=")
                        + std::to_string(params.batch_sz) + std::string(" -DH=")
                        + std::to_string(params.in_height) + std::string(" -DW=")
                        + std::to_string(params.in_width) + std::string(" -DC=")
                        + std::to_string(params.n_inputs) + std::string(" -DK=")
                        + std::to_string(params.n_outputs) + std::string(" -DSTRIDE=")
                        + std::to_string(params.kernel_stride0) + std::string(" -DGLOBAL_SPLITU=")
                        + std::to_string(conv11_param.params.global_split)
                        + std::string(" -DPER_ITER_STRIDE=")
                        + std::to_string(conv11_param.params.stride_per_iter)
                        + std::string(" -DKERNEL_METHOD=") + std::to_string(conv11_param.kernel_method)
                        + std::string(" -DTILE_COL=") + std::to_string(conv11_param.params.tile_col)
                        + std::string(" -DTILE_ROW=") + std::to_string(conv11_param.params.tile_row)
                        + std::string(" -DPER_WI_TILE_ROW=")
                        + std::to_string(conv11_param.params.wi_per_tile_row)
                        + std::string(" -DPER_WI_TILE_COL=")
                        + std::to_string(conv11_param.params.wi_per_tile_col)
                        + std::string(" -DBRANCH=") + std::to_string(conv11_param.params.code_branch)
                        + std::string(" -DMETHOD=") + std::to_string(conv11_param.params.code_method);
                } else {
                    kernelInfo.comp_options =
                        std::string(" -DBIAS=") + std::to_string(params.bias)
                        + std::string(" -DN=") + std::to_string(params.batch_sz)
                        + std::string(" -DSTRIDE=")
                        + std::to_string(params.kernel_stride0) + std::string(" -DGLOBAL_SPLITU=")
                        + std::to_string(conv11_param.params.global_split)
                        + std::string(" -DPER_ITER_STRIDE=")
                        + std::to_string(conv11_param.params.stride_per_iter)
                        + std::string(" -DKERNEL_METHOD=") + std::to_string(conv11_param.kernel_method)
                        + std::string(" -DTILE_COL=") + std::to_string(conv11_param.params.tile_col)
                        + std::string(" -DTILE_ROW=") + std::to_string(conv11_param.params.tile_row)
                        + std::string(" -DPER_WI_TILE_ROW=")
                        + std::to_string(conv11_param.params.wi_per_tile_row)
                        + std::string(" -DPER_WI_TILE_COL=")
                        + std::to_string(conv11_param.params.wi_per_tile_col)
                        + std::string(" -DBRANCH=") + std::to_string(conv11_param.params.code_branch)
                        + std::string(" -DMETHOD=") + std::to_string(conv11_param.params.code_method);
                }

                int wg_in = (params.out_height * params.out_width + conv11_param.params.tile_col - 1)
                            / conv11_param.params.tile_col;
                int wg_wei = (params.n_outputs + conv11_param.params.tile_row - 1)
                             / conv11_param.params.tile_row;

                kernelInfo.l_wk = {256, 1, 1};
                kernelInfo.g_wk = {256 * wg_in* wg_wei * conv11_param.params.global_split, 1, 1};
            } else if (conv11_param.kernel_method <= 3) {
                kernelInfo.kernel_name = "conv1x1_act_pool";

                if (params.bias) {
                    kernelInfo.comp_options = std::string(" -DBIAS ") + std::string(" -DN=")
                                              + std::to_string(params.batch_sz) + std::string(" -DH=")
                                              + std::to_string(params.in_height) + std::string(" -DW=")
                                              + std::to_string(params.in_width) + std::string(" -DC=")
                                              + std::to_string(params.n_inputs) + std::string(" -DK=")
                                              + std::to_string(params.n_outputs)
                                              + std::string(" -DKERNEL_METHOD=") + std::to_string(conv11_param.kernel_method);
                } else {
                    kernelInfo.comp_options = std::string(" -DN=") + std::to_string(params.batch_sz)
                                              + std::string(" -DH=") + std::to_string(params.in_height)
                                              + std::string(" -DW=") + std::to_string(params.in_width)
                                              + std::string(" -DC=") + std::to_string(params.n_inputs)
                                              + std::string(" -DK=") + std::to_string(params.n_outputs)
                                              + std::string(" -DKERNEL_METHOD=") + std::to_string(conv11_param.kernel_method);
                }

                kernelInfo.l_wk = {1024, 1, 1};
                kernelInfo.g_wk = {1024 * 64, 1, 1};
            } else if (conv11_param.kernel_method == 4) {
                if (conv11_param.params.tile_col != 0) {
                    kernelInfo.kernel_name = "conv1x1_act";

                    if (cc._usemacro) {
                        kernelInfo.comp_options =
                            std::string(" -DMACRO")
                            + std::string(" -DBIAS=") + std::to_string(params.bias) + std::string(" -DN=")
                            + std::to_string(params.batch_sz) + std::string(" -DH=")
                            + std::to_string(params.in_height) + std::string(" -DW=")
                            + std::to_string(params.in_width) + std::string(" -DC=")
                            + std::to_string(params.n_inputs) + std::string(" -DK=")
                            + std::to_string(params.n_outputs) + std::string(" -DSTRIDE=")
                            + std::to_string(params.kernel_stride0) + std::string(" -DGLOBAL_SPLITU=")
                            + std::to_string(conv11_param.params.global_split)
                            + std::string(" -DPER_ITER_STRIDE=")
                            + std::to_string(conv11_param.params.stride_per_iter)
                            + std::string(" -DKERNEL_METHOD=") + std::to_string(conv11_param.kernel_method)
                            + std::string(" -DTILE_COL=") + std::to_string(conv11_param.params.tile_col)
                            + std::string(" -DTILE_ROW=") + std::to_string(conv11_param.params.tile_row)
                            + std::string(" -DPER_WI_TILE_ROW=")
                            + std::to_string(conv11_param.params.wi_per_tile_row)
                            + std::string(" -DPER_WI_TILE_COL=")
                            + std::to_string(conv11_param.params.wi_per_tile_col)
                            + std::string(" -DBRANCH=") + std::to_string(conv11_param.params.code_branch)
                            + std::string(" -DMETHOD=") + std::to_string(conv11_param.params.code_method);
                    } else {
                        kernelInfo.comp_options =
                            std::string(" -DBIAS=") + std::to_string(params.bias)
                            + std::string(" -DN=") + std::to_string(params.batch_sz)
                            + std::string(" -DSTRIDE=")
                            + std::to_string(params.kernel_stride0) + std::string(" -DGLOBAL_SPLITU=")
                            + std::to_string(conv11_param.params.global_split)
                            + std::string(" -DPER_ITER_STRIDE=")
                            + std::to_string(conv11_param.params.stride_per_iter)
                            + std::string(" -DKERNEL_METHOD=") + std::to_string(conv11_param.kernel_method)
                            + std::string(" -DTILE_COL=") + std::to_string(conv11_param.params.tile_col)
                            + std::string(" -DTILE_ROW=") + std::to_string(conv11_param.params.tile_row)
                            + std::string(" -DPER_WI_TILE_ROW=")
                            + std::to_string(conv11_param.params.wi_per_tile_row)
                            + std::string(" -DPER_WI_TILE_COL=")
                            + std::to_string(conv11_param.params.wi_per_tile_col)
                            + std::string(" -DBRANCH=") + std::to_string(conv11_param.params.code_branch)
                            + std::string(" -DMETHOD=") + std::to_string(conv11_param.params.code_method);
                    }

                    int wg_in = (params.out_height * params.out_width + conv11_param.params.tile_col - 1)
                                / conv11_param.params.tile_col;
                    int wg_wei = (params.n_outputs + conv11_param.params.tile_row - 1)
                                 / conv11_param.params.tile_row;

                    kernelInfo.l_wk = {256, 1, 1};
                    kernelInfo.g_wk = {256 * wg_in* wg_wei * conv11_param.params.global_split, 1, 1};
                } else {
                    int stride_per_iter = 16;

                    int tile_col;
                    int tile_row;
                    int wg_in;
                    int wg_wei;

                    for (int i = 0; i < ARG_SIZE; i++) {

                        tile_col = args[i].tile_col;
                        tile_row = args[i].tile_row;

                        wg_in =
                            (params.in_height * params.in_width / (params.kernel_stride0 * params.kernel_stride1) + tile_col -
                             1) /
                            tile_col;
                        wg_wei =
                            (params.n_outputs + tile_row - 1) / tile_row;

                        int wl = wg_in * wg_wei * 8;

                        if (wl > 128) {
                            break;
                        }
                    }

                    int global_split = 1;

                    for (; wg_in * wg_wei * global_split < 128 && global_split < 8; global_split *= 2) {
                    }

                    if (params.n_inputs % (16 * global_split) != 0 || params.n_outputs % tile_row != 0) {
                        global_split = 1;
                    }

                    int wi_per_tile_col = tile_col / 16;
                    int wi_per_tile_row = tile_row / 16;



                    kernelInfo.kernel_file = "Conv1x1.cl";
                    kernelInfo.kernel_name = "conv1x1_act";

                    if (cc._usemacro) {
                        kernelInfo.comp_options =
                            std::string(" -DMACRO") +
                            std::string(" -DBIAS=") + std::to_string(params.bias) +
                            std::string(" -DN=") + std::to_string(params.batch_sz) +
                            std::string(" -DH=") + std::to_string(params.in_height) +
                            std::string(" -DW=") + std::to_string(params.in_width) +
                            std::string(" -DC=") + std::to_string(params.n_inputs) +
                            std::string(" -DK=") + std::to_string(params.n_outputs) +
                            std::string(" -DSTRIDE=") + std::to_string(params.kernel_stride0) +
                            std::string(" -DGLOBAL_SPLITU=") + std::to_string(global_split) +
                            std::string(" -DPER_ITER_STRIDE=") + std::to_string(stride_per_iter) +
                            std::string(" -DTILE_COL=") + std::to_string(tile_col) +
                            std::string(" -DTILE_ROW=") + std::to_string(tile_row) +
                            std::string(" -DPER_WI_TILE_ROW=") + std::to_string(wi_per_tile_row) +
                            std::string(" -DPER_WI_TILE_COL=") + std::to_string(wi_per_tile_col) +
                            std::string(" -DBRANCH=1") +
                            std::string(" -DKERNEL_METHOD=") + std::to_string(conv11_param.kernel_method);
                    } else {
                        kernelInfo.comp_options =
                            std::string(" -DBIAS=") + std::to_string(params.bias) +
                            std::string(" -DN=") + std::to_string(params.batch_sz) +
                            std::string(" -DSTRIDE=") + std::to_string(params.kernel_stride0) +
                            std::string(" -DGLOBAL_SPLITU=") + std::to_string(global_split) +
                            std::string(" -DPER_ITER_STRIDE=") + std::to_string(stride_per_iter) +
                            std::string(" -DTILE_COL=") + std::to_string(tile_col) +
                            std::string(" -DTILE_ROW=") + std::to_string(tile_row) +
                            std::string(" -DPER_WI_TILE_ROW=") + std::to_string(wi_per_tile_row) +
                            std::string(" -DPER_WI_TILE_COL=") + std::to_string(wi_per_tile_col) +
                            std::string(" -DBRANCH=1") +
                            std::string(" -DKERNEL_METHOD=") + std::to_string(conv11_param.kernel_method);
                    }

                    kernelInfo.l_wk        = {256, 1, 1};
                    kernelInfo.g_wk        = {256 * wg_in* wg_wei * global_split, 1, 1};
                }
            }
        } else if (conv11_param.kernel_name == "Conv1x1FC.cl") {
            kernelInfo.isMIOpenKernel = true;
            kernelInfo.kernel_name = "InnerProduct";

            if (params.bias) {
                kernelInfo.comp_options = std::string(" -DBIAS ") + std::string(" -DWIDTH=")
                                          + std::to_string(params.n_inputs)
                                          + std::string(" -DMACRO -DNO_SLOPE")
                                          + std::string(" -DOUTPUT=") + std::to_string(params.n_outputs)
                                          + std::string(" -DKERNEL_METHOD=") + std::to_string(conv11_param.kernel_method)
                                          + std::string(" -DN=") + std::to_string(params.batch_sz);
            } else {
                kernelInfo.comp_options =
                    std::string(" -DWIDTH=") + std::to_string(params.n_inputs)
                    + std::string(" -DOUTPUT=") + std::to_string(params.n_outputs)
                    + std::string(" -DMACRO -DNO_SLOPE")
                    + std::string(" -DKERNEL_METHOD=") + std::to_string(conv11_param.kernel_method)
                    + std::string(" -DN=") + std::to_string(params.batch_sz);
            }

            kernelInfo.l_wk = {64, 1, 1};
            kernelInfo.g_wk = {64 * 64 * 16, 1, 1};

        } else if (conv11_param.kernel_name == "xGemm") {
            kernelInfo.kernel_name = "xGemm";
        } else if (conv11_param.kernel_name == "ConvFwd1x1_7x7x512x2048x1.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {262144, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_7x7x2048x512x1.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {128, 1, 1};
            kernelInfo.g_wk = {524288, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_14x14x256x1024x1.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {128, 1, 1};
            kernelInfo.g_wk = {32768, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_14x14x1024x256x1.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {64, 1, 1};
            kernelInfo.g_wk = {32768, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_28x28x128x512x1.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {512, 1, 1};
            kernelInfo.g_wk = {32768, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_28x28x512x128x1.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {512, 1, 1};
            kernelInfo.g_wk = {32768, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_56x56x64x64x1.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {512, 1, 1};
            kernelInfo.g_wk = {28672, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_56x56x64x256x1.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {106496, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_56x56x256x64x1.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {26624, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_7x7x512x2048x2.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {65536, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_7x7x2048x512x2.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {65536, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_14x14x256x1024x2.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {64, 1, 1};
            kernelInfo.g_wk = {28672, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_14x14x1024x256x2.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {64, 1, 1};
            kernelInfo.g_wk = {28672, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_28x28x128x512x2.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {512, 1, 1};
            kernelInfo.g_wk = {65536, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_28x28x512x128x2.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {128, 1, 1};
            kernelInfo.g_wk = {26624, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_56x56x64x64x2.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {51200, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_56x56x64x256x2.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {102400, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_56x56x256x64x2.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {51200, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_7x7x512x2048x4.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {128, 1, 1};
            kernelInfo.g_wk = {32768, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_7x7x2048x512x4.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {128, 1, 1};
            kernelInfo.g_wk = {32768, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_14x14x256x1024x4.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {64, 1, 1};
            kernelInfo.g_wk = {53248, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_14x14x1024x256x4.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {512, 1, 1};
            kernelInfo.g_wk = {32768, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_28x28x128x512x4.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {64, 1, 1};
            kernelInfo.g_wk = {100352, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_28x28x512x128x4.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {53248, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_56x56x64x64x4.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {50176, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_56x56x64x256x4.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {128, 1, 1};
            kernelInfo.g_wk = {200704, 1, 1};
        } else if (conv11_param.kernel_name == "ConvFwd1x1_56x56x256x64x4.s") {
            kernelInfo.kernel_name = "ConvFwd1x1";
            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {100352, 1, 1};
        }

        kernelInfo.isMIOpenKernel = true;
        result.construction_params.push_back(kernelInfo);
    } else {
        result.status = miopenStatusInternalError;
        ALOGE("can NOT get solution");
    }

    // Start to do pooling...
    if (params.has_pooling
            && (kernelInfo.kernel_name != "conv1x1_act_pool")
            && (kernelInfo.kernel_name != "xGemm")) {
        addPoolingKernel(params, result);
    }

    return result;
}

template <class Solver>
static void ExpandGetSolution(ConvSolution& kernel_search_result, const ConvolutionContext& params,
                              const LegacyPerformanceConfig& result, Solver&& s) {
    if (!kernel_search_result.Succeeded()) { // once
        if (s.IsApplicable(params)) {
            if (s.IsValidPerformanceConfig(params, result)) {
                kernel_search_result = s.GetSolution(params, result);
            }
        }
    }
}

template <class Solver, class... Solvers>
static void ExpandGetSolution(ConvSolution& kernel_search_result, const ConvolutionContext& params,
                              const LegacyPerformanceConfig& result, Solver&& solver, Solvers&& ... solvers) {
    ExpandGetSolution(kernel_search_result, result, solver);
    ExpandGetSolution(kernel_search_result, result, solvers...);
}

template <class... Solvers>
static int MeasureLoop(Handle* profile_h,
                       Data_t bot_ocl_buf,
                       Data_t top_ocl_buf,
                       Data_t wei_ocl_buf,
                       Data_t bias_ocl_buf,
                       double& processing_time,
                       const ConvolutionContext& params,
                       const LegacyPerformanceConfig& result) {
    ConvSolution kernel_search_result {miopenStatusNotInitialized};

#if(__cplusplus >= 201402L)
    miopen::each_args(
    [&](auto s) {
        if (!kernel_search_result.Succeeded()) { // once
            if (s.IsApplicable(params)) {
                if (s.IsValidPerformanceConfig(params, result)) {
                    kernel_search_result = s.GetSolution(params, result);
                }
            }
        }
    },
    Solvers {} ...);
#else
    ExpandGetSolution(kernel_search_result, params, result, Solvers {} ...);
#endif

    if (!kernel_search_result.Succeeded()) {
        return 1;
    }

    MIOPEN_LOG_I2("Trying " << result);
    const auto kernel_params     = kernel_search_result.construction_params[0];
    std::string compiler_options = params.general_compile_options + kernel_params.comp_options;

    // Creating OCLKernel obj
    try {

        float padding_value = 0;

        if (profile_h) {
            processing_time = std::numeric_limits<float>::max();



            std::cout << "kernel name:" << kernel_params.kernel_name << std::endl;

            if (kernel_params.kernel_name == "conv1x1_act") {
                auto k = profile_h->AddKernel("",
                                              "",
                                              kernel_params.kernel_file,
                                              kernel_params.kernel_name,
                                              kernel_params.l_wk,
                                              kernel_params.g_wk,
                                              compiler_options);

                if (params.bias) {
                    k(wei_ocl_buf, bot_ocl_buf, bias_ocl_buf, top_ocl_buf, params.negative_slope, params.n_inputs,
                      params.in_height, params.in_width, params.n_outputs);
                } else {
                    k(wei_ocl_buf, bot_ocl_buf, top_ocl_buf, params.negative_slope, params.n_inputs, params.in_height,
                      params.in_width, params.n_outputs);
                }
            } else if (kernel_params.kernel_name == "InnerProduct") {
                auto k = profile_h->AddKernel("",
                                              "",
                                              kernel_params.kernel_file,
                                              kernel_params.kernel_name,
                                              kernel_params.l_wk,
                                              kernel_params.g_wk,
                                              compiler_options);

                if (params.bias) {
                    if (params.has_active) {
                        k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, params.negative_slope);
                    } else {
                        k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf);
                    }
                } else {
                    if (params.has_active) {
                        k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, params.negative_slope);
                    } else {
                        k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf);
                    }
                }
            } else {
                auto k = profile_h->AddKernel("",
                                              "",
                                              kernel_params.kernel_file,
                                              kernel_params.kernel_name,
                                              kernel_params.l_wk,
                                              kernel_params.g_wk,
                                              "");

                size_t slot_sz = 4096;

                std::vector<float> slot_sys_buf(slot_sz);
                auto slot_ocl_buf = profile_h->Write(slot_sys_buf);

                size_t bias_sz = params.n_outputs;

                std::vector<float> bias_sys_buf(bias_sz);
                auto tmp_bias_ocl_buf = profile_h->Write(bias_sys_buf);

                if (params.has_active) {
                    k(bot_ocl_buf, wei_ocl_buf, tmp_bias_ocl_buf.get(), top_ocl_buf, slot_ocl_buf.get(),
                      params.negative_slope);
                } else {
                    k(bot_ocl_buf, wei_ocl_buf, tmp_bias_ocl_buf.get(), top_ocl_buf, slot_ocl_buf.get(), 1.0f);
                }
            }

            processing_time = profile_h->GetKernelTime();
        }
    }

    catch (miopen::Exception& ex) {
        MIOPEN_LOG_E("MeasureLoop failed for: " << ex.what());
        return -1;
    }

    //MIOPEN_LOG_I2("\t\t\t\t" << processing_time);
    std::cout << "\t\t\t\t" << processing_time << std::endl;
    return 0;
}


LegacyPerformanceConfig
ConvOclDirectFwd1x1AMD::SearchForMeasureOnce(const ConvolutionContext& params) const {
    LegacyPerformanceConfig result;

    miopen::Handle profile_h;
    double processing_time = std::numeric_limits<double>::max();
    double min_proc_time = std::numeric_limits<double>::max();



    // allocate tem input/output buffers
    size_t bot_sz = params.bot_sz / sizeof(float);
    std::vector<float> bot_sys_buf(bot_sz);

    for (int i = 0; i < bot_sz; i++) {
        bot_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
    }

    auto bot_ocl_buf = profile_h.Write(bot_sys_buf);

    size_t top_sz = params.top_sz / sizeof(float);
    std::vector<float> top_sys_buf(top_sz);

    auto top_ocl_buf = profile_h.Write(top_sys_buf);

    std::vector<float> random_top_sys_buf(top_sz);

    for (int i = 0; i < top_sz; i++) {
        random_top_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
    }

    size_t weights_sz = params.weights_sz / sizeof(float);
    std::vector<float> wei_sys_buf(weights_sz);

    std::cout << "top_sz:" << top_sz << " bot_sz:" << bot_sz << " weights_sz:" << weights_sz <<
              std::endl;

    for (int i = 0; i < weights_sz; i++) {
        wei_sys_buf[i] = static_cast<float>((rand() * (1.0 / RAND_MAX) - 0.5) * 0.001);
    }

    auto wei_ocl_buf = profile_h.Write(wei_sys_buf);

    std::vector<float> bias_sys_buf;
    miopen::Allocator::ManageDataPtr bias_ocl_buf = nullptr;

    if (params.bias != 0) {
        size_t bias_sz = params.bias_sz / sizeof(float);
        bias_sys_buf   = std::vector<float>(bias_sz);

        for (int i = 0; i < bias_sz; i++) {
            bias_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
        }

        bias_ocl_buf = profile_h.Write(bias_sys_buf);
    }

    // randomize output
    profile_h.WriteTo(reinterpret_cast<const void*>(random_top_sys_buf.data()),
                      top_ocl_buf,
                      random_top_sys_buf.size() * sizeof(float));

    // enable profiling for the handle for benchmarking
    profile_h.EnableProfiling(true);
    const auto ret = MeasureLoop<ConvOclDirectFwd1x1AMD>(&profile_h,
                     bot_ocl_buf.get(),
                     top_ocl_buf.get(),
                     wei_ocl_buf.get(),
                     bias_ocl_buf.get(),
                     processing_time,
                     params,
                     result);

    if (min_proc_time > processing_time) {
        min_proc_time = processing_time;
        MIOPEN_LOG_I2("processing_time = " << processing_time << ", result = " << result);
    }

    std::cout << std::endl << "=Score: " << min_proc_time << std::endl;

    result.grp_tile0       = 0;
    result.grp_tile1       = 0;
    result.in_tile0        = 0;
    result.in_tile1        = 0;
    result.out_pix_tile0   = 0;
    result.out_pix_tile1   = 0;
    result.n_out_pix_tiles = 0;
    result.n_in_data_tiles = 0;
    result.n_stacks        = 0;
    result.min_proc_time   = min_proc_time;

    profile_h.EnableProfiling(false);
    return result;
}

} // namespace solver
} // namespace miopen
