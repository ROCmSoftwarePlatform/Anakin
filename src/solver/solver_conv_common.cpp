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
#include "miopen/solver_conv_common.hpp"

namespace miopen {

bool ConvCommon::getKernelInfo(const ConvolutionContext& params, Conv1x1Type& mType) {
    int dev;
    const auto dev_name = params.GetStream().GetDeviceName();

    if (dev_name == "gfx803") {
        dev = GFX803;
    } else if (dev_name == "gfx900") {
        dev = GFX900;
    }

    for (int i = 0; i < conv1x1type.size(); i++) {
        if (params.has_pooling && conv1x1type[i].fusion_pooling
                && params.poolingContext.pooling_type == MLO_POOLING_OP_MAX
                && params.poolingContext.kernel_size1 == 2
                && params.poolingContext.kernel_size0 == 2
                && params.poolingContext.pad1 == 0
                && params.poolingContext.pad0 == 0
                && params.poolingContext.kernel_stride1 == 2
                && params.poolingContext.kernel_stride0 == 2
                && conv1x1type[i].dev == dev && conv1x1type[i].batch == params.batch_sz
                && conv1x1type[i].stride == params.kernel_stride0 && conv1x1type[i].channel == params.n_inputs
                && conv1x1type[i].width == params.in_width && conv1x1type[i].output_num == params.n_outputs) {

            mType = conv1x1type[i];
            ALOGD("Got kernel:" << mType.kernel_name << "!!");
            return true;
        } else if (!conv1x1type[i].fusion_pooling && conv1x1type[i].dev == dev
                   && conv1x1type[i].batch == params.batch_sz
                   && conv1x1type[i].stride == params.kernel_stride0 && (conv1x1type[i].channel == params.n_inputs
                           || conv1x1type[i].channel == 0)
                   && conv1x1type[i].width == params.in_width && conv1x1type[i].output_num == params.n_outputs
                   && params.n_inputs % 16 == 0) {

            mType = conv1x1type[i];
            ALOGD("Got kernel:" << mType.kernel_name << "!!");
            return true;
        }

        if (conv1x1type[i].dev == dev
                && ((conv1x1type[i].channel == params.n_inputs && params.n_inputs == 1024)
                    || (conv1x1type[i].channel == params.n_inputs && params.n_inputs == 1280))
                && (conv1x1type[i].output_num == params.n_outputs && params.n_outputs == 1000)
                && (conv1x1type[i].stride == params.kernel_stride0 && params.kernel_stride0 == 1)
                && (conv1x1type[i].width == params.in_width && params.in_width == 1)
                && conv1x1type[i].batch >= params.batch_sz) {
            mType = conv1x1type[i];
            ALOGD("Got kernel:" << mType.kernel_name << "!!");
            return true;
        }
    }

    if (params.batch_sz < 32 && ((params.batch_sz == 1 && params.kernel_stride0 == 1)
                                 || (params.in_width <= 14 && params.kernel_stride0 == 1)
                                 || (params.kernel_stride0 == 2))) {
        mType.kernel_name = "xGemm";
        ALOGD("Got kernel:" << mType.kernel_name << "!!");
        return true;
    }

    return false;
}

} // namespace miopen
