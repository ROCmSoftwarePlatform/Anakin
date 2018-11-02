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
#include "miopen/solver_conv_common.hpp"

namespace miopen {
namespace solver {

bool ConvOclDirectFwd1x1AMD::IsApplicable(const ConvolutionContext& params) const {
    bool result =
            params.direction.IsForward() && (params.kernel_size0 == 1 && params.kernel_size1 == 1)
            && (params.batch_sz <= 2) && (params.kernel_stride0 <= 2)
            && (params.pad0 == 0 && params.pad1 == 0)
            && (params.n_inputs == 16 || params.n_inputs == 24 || params.n_inputs == 32
                || params.n_inputs == 64 || params.n_inputs == 96 || params.n_inputs == 128
                || params.n_inputs == 144 || params.n_inputs == 160 || params.n_inputs == 192
                || params.n_inputs == 256 || params.n_inputs == 320 || params.n_inputs == 384
                || params.n_inputs == 512 || params.n_inputs == 576 || params.n_inputs == 960
                || params.n_inputs == 1024 || params.n_inputs == 1280 || params.n_inputs == 2048)
            && ((params.in_height == params.in_width)
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
        Conv1x1Type* conv11_param = cc.getKernelInfo(
                dev,
                params.batch_sz,
                params.kernel_stride0,
                params.n_inputs,
                params.in_width,
                params.n_outputs);

        if (conv11_param == NULL) {
            result = false;
        } else {
            ALOGD("ConvOclDirectFwd1x1AMD::IsApplicable result:" << result << " kernel name="
                                                                 << conv11_param->kernel_name);
        }
    }
    return result;
}

ConvSolution ConvOclDirectFwd1x1AMD::GetSolution(
        const ConvolutionContext& params,
        const LegacyPerformanceConfig& searched_params) const {
    ConvSolution result;
    int dev;
    // searched_params.CopyTo(result);

    const auto dev_name = params.GetStream().GetDeviceName();
    if (dev_name == "gfx803") {
        dev = GFX803;
    } else if (dev_name == "gfx900") {
        dev = GFX900;
    }

    ConvCommon cc;
    Conv1x1Type* conv11_param = cc.getKernelInfo(
            dev,
            params.batch_sz,
            params.kernel_stride0,
            params.n_inputs,
            params.in_width,
            params.n_outputs);

    KernelInfo kernelInfo;
    if (conv11_param != NULL) {
        kernelInfo.kernel_file = conv11_param->kernel_name;
        if (conv11_param->kernel_name == "Conv1x1Atomic.cl") {
            kernelInfo.kernel_name = "conv1x1_act";

            kernelInfo.comp_options =
                    std::string(" -DBIAS=") + std::to_string(params.bias) + std::string(" -DN=")
                    + std::to_string(params.batch_sz) + std::string(" -DH=")
                    + std::to_string(params.in_height) + std::string(" -DW=")
                    + std::to_string(params.in_width) + std::string(" -DC=")
                    + std::to_string(params.n_inputs) + std::string(" -DK=")
                    + std::to_string(params.n_outputs) + std::string(" -DSTRIDE=")
                    + std::to_string(params.kernel_stride0) + std::string(" -DGLOBAL_SPLITU=")
                    + std::to_string(conv11_param->params.global_split)
                    + std::string(" -DPER_ITER_STRIDE=")
                    + std::to_string(conv11_param->params.stride_per_iter)
                    + std::string(" -DTILE_COL=") + std::to_string(conv11_param->params.tile_col)
                    + std::string(" -DTILE_ROW=") + std::to_string(conv11_param->params.tile_row)
                    + std::string(" -DPER_WI_TILE_ROW=")
                    + std::to_string(conv11_param->params.wi_per_tile_col)
                    + std::string(" -DPER_WI_TILE_COL=")
                    + std::to_string(conv11_param->params.wi_per_tile_row)
                    + std::string(" -DBRANCH=") + std::to_string(conv11_param->params.code_branch)
                    + std::string(" -DMETHOD=") + std::to_string(conv11_param->params.code_method);

            int wg_in = (params.out_height * params.out_width + conv11_param->params.tile_col - 1)
                        / conv11_param->params.tile_col;
            int wg_wei = (params.n_outputs + conv11_param->params.tile_row - 1)
                         / conv11_param->params.tile_row;

            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {256 * wg_in * wg_wei * conv11_param->params.global_split, 1, 1};
        } else if (conv11_param->kernel_name == "Conv1x1FC7.cl") {
            kernelInfo.kernel_name = "InnerProduct";
            if (params.bias) {
                kernelInfo.comp_options = std::string(" -DBIAS ") + std::string(" -DSTRIDE=")
                                          + std::to_string(params.n_inputs);
            } else {
                kernelInfo.comp_options =
                        std::string(" -DSTRIDE=") + std::to_string(params.n_inputs);
            }

            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {256 * 64 * 1, 1, 1};
        } else if (conv11_param->kernel_name == "Conv1x1CXH7W7K160.cl") {
            kernelInfo.kernel_name = "conv1x1_act";

            if (params.bias) {
                kernelInfo.comp_options = std::string(" -DBIAS ") + std::string(" -DN=")
                                          + std::to_string(params.batch_sz) + std::string(" -DH=")
                                          + std::to_string(params.in_height) + std::string(" -DW=")
                                          + std::to_string(params.in_width) + std::string(" -DC=")
                                          + std::to_string(params.n_inputs) + std::string(" -DK=")
                                          + std::to_string(params.n_outputs);
            } else {
                kernelInfo.comp_options = std::string(" -DN=") + std::to_string(params.batch_sz)
                                          + std::string(" -DH=") + std::to_string(params.in_height)
                                          + std::string(" -DW=") + std::to_string(params.in_width)
                                          + std::string(" -DC=") + std::to_string(params.n_inputs)
                                          + std::string(" -DK=") + std::to_string(params.n_outputs);
            }

            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {256 * 40 * 4, 1, 1};
        } else if (conv11_param->kernel_name == "Conv1x1CXH7W7K320.cl") {
            kernelInfo.kernel_name = "conv1x1_act";

            if (params.bias) {
                kernelInfo.comp_options = std::string(" -DBIAS ") + std::string(" -DN=")
                                          + std::to_string(params.batch_sz) + std::string(" -DH=")
                                          + std::to_string(params.in_height) + std::string(" -DW=")
                                          + std::to_string(params.in_width) + std::string(" -DC=")
                                          + std::to_string(params.n_inputs) + std::string(" -DK=")
                                          + std::to_string(params.n_outputs);
            } else {
                kernelInfo.comp_options = std::string(" -DN=") + std::to_string(params.batch_sz)
                                          + std::string(" -DH=") + std::to_string(params.in_height)
                                          + std::string(" -DW=") + std::to_string(params.in_width)
                                          + std::string(" -DC=") + std::to_string(params.n_inputs)
                                          + std::string(" -DK=") + std::to_string(params.n_outputs);
            }

            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {256 * 40 * 4, 1, 1};
        } else if (conv11_param->kernel_name == "Conv1x1CXH14W14K96.cl") {
            kernelInfo.kernel_name = "conv1x1_act";

            if (params.bias) {
                kernelInfo.comp_options = std::string(" -DBIAS ") + std::string(" -DN=")
                                          + std::to_string(params.batch_sz) + std::string(" -DH=")
                                          + std::to_string(params.in_height) + std::string(" -DW=")
                                          + std::to_string(params.in_width) + std::string(" -DC=")
                                          + std::to_string(params.n_inputs) + std::string(" -DK=")
                                          + std::to_string(params.n_outputs);
            } else {
                kernelInfo.comp_options = std::string(" -DN=") + std::to_string(params.batch_sz)
                                          + std::string(" -DH=") + std::to_string(params.in_height)
                                          + std::string(" -DW=") + std::to_string(params.in_width)
                                          + std::string(" -DC=") + std::to_string(params.n_inputs)
                                          + std::string(" -DK=") + std::to_string(params.n_outputs);
            }

            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {256 * 39 * 4, 1, 1};
        } else if (conv11_param->kernel_name == "Conv1x1C256H56W56K512S2.cl") {
            kernelInfo.kernel_name = "conv1x1_act";

            if (params.bias) {
                kernelInfo.comp_options = std::string(" -DBIAS ") + std::string(" -DN=")
                                          + std::to_string(params.batch_sz) + std::string(" -DH=")
                                          + std::to_string(params.in_height) + std::string(" -DW=")
                                          + std::to_string(params.in_width) + std::string(" -DC=")
                                          + std::to_string(params.n_inputs) + std::string(" -DK=")
                                          + std::to_string(params.n_outputs);
            } else {
                kernelInfo.comp_options = std::string(" -DN=") + std::to_string(params.batch_sz)
                                          + std::string(" -DH=") + std::to_string(params.in_height)
                                          + std::string(" -DW=") + std::to_string(params.in_width)
                                          + std::string(" -DC=") + std::to_string(params.n_inputs)
                                          + std::string(" -DK=") + std::to_string(params.n_outputs);
            }

            kernelInfo.l_wk = {256, 1, 1};
            kernelInfo.g_wk = {256 * 200, 1, 1};
        } else if (conv11_param->kernel_name == "Conv1x1C320H7W7K1280Pool.cl") {
            kernelInfo.kernel_name = "conv1x1_act_pool";

            if (params.bias) {
                kernelInfo.comp_options = std::string(" -DBIAS ") + std::string(" -DN=")
                                          + std::to_string(params.batch_sz) + std::string(" -DH=")
                                          + std::to_string(params.in_height) + std::string(" -DW=")
                                          + std::to_string(params.in_width) + std::string(" -DC=")
                                          + std::to_string(params.n_inputs) + std::string(" -DK=")
                                          + std::to_string(params.n_outputs);
            } else {
                kernelInfo.comp_options = std::string(" -DN=") + std::to_string(params.batch_sz)
                                          + std::string(" -DH=") + std::to_string(params.in_height)
                                          + std::string(" -DW=") + std::to_string(params.in_width)
                                          + std::string(" -DC=") + std::to_string(params.n_inputs)
                                          + std::string(" -DK=") + std::to_string(params.n_outputs);
            }

            kernelInfo.l_wk = {1024, 1, 1};
            kernelInfo.g_wk = {1024 * 64, 1, 1};
        }
        kernelInfo.isMIOpenKernel = false;
        result.construction_params.push_back(kernelInfo);
    } else {
        ALOGE("can NOT get solution");
    }

    return result;
}
} // namespace solver
} // namespace miopen
