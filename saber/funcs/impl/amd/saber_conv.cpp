/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "saber/funcs/impl/amd/include/saber_conv.h"
#include "saber/funcs/conv.h"
#include "third-party/miopen/include/miopen/solver.hpp"
#include "saber/core/impl/amd/utils/amd_file_utils.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;
typedef Tensor<AMD> TensorDf4;

template <DataType OpDtype>
SaberStatus SaberConv2D<AMD, OpDtype>::init(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        ConvParam<AMD>& param,
        Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberConv2D<AMD, OpDtype>::create(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        ConvParam<AMD>& param,
        Context<AMD>& ctx) {
    ALOGD("create");
    this->_ctx = &ctx;
    KernelInfo kernelInfo;
    AMDKernelPtr kptr;
    _kernels_ptr.clear();

    ALOGD("num=" << inputs[0]->num() << " channel=" << inputs[0]->channel()
                 << " height=" << inputs[0]->height() << " width=" << inputs[0]->width());

    ALOGD("stride_h=" << param.stride_h << " stride_w=" << param.stride_w << " height="
                      << param.weight()->height() << " width=" << param.weight()->width()
                      << " group=" << param.group << " dilation_h=" << param.dilation_h
                      << " dilation_w=" << param.dilation_w << " pad_h=" << param.pad_h << " pad_w"
                      << param.pad_w << " alpha=" << param.alpha << " beta=" << param.beta);
    if (param.activation_param.has_active)
        ALOGD("activation.active=" << param.activation_param.active);
    if (param.bias()->valid_size() > 0)
        ALOGD("bias size=" << param.bias()->size());

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; // anakin device id to AMD device
    cl_device_id device = dev.get_device();
    cl_context context  = dev.get_context();
    miopen::ConvolutionContext convContext;
    convContext.direction.Set(1);
    convContext.general_compile_options += "";
    convContext.n_inputs                = inputs[0]->channel();
    convContext.in_height               = inputs[0]->height();
    convContext.in_width                = inputs[0]->width();
    convContext.kernel_size0            = param.weight()->width();
    convContext.kernel_size1            = param.weight()->height();
    convContext.n_outputs               = param.weight()->num();
    convContext.out_height              = outputs[0]->height();
    convContext.out_width               = outputs[0]->width();
    convContext.batch_sz                = inputs[0]->num();
    convContext.pad0                    = param.pad_w;
    convContext.pad1                    = param.pad_h;
    convContext.kernel_stride0          = param.stride_w;
    convContext.kernel_stride1          = param.stride_h;
    convContext.kernel_dilation0        = param.dilation_w;
    convContext.kernel_dilation1        = param.dilation_h;
    convContext.bias                    = (param.bias()->size() > 0) ? 1 : 0;
    convContext.float_size              = 32;
    convContext.in_stride               = inputs[0]->get_stride()[2];
    convContext.out_stride              = outputs[0]->get_stride()[2];
    convContext.in_channel_stride       = convContext.in_stride * convContext.in_height;
    convContext.in_batch_stride         = convContext.in_channel_stride * convContext.n_inputs;
    convContext.out_channel_stride      = convContext.out_stride * convContext.out_height;
    convContext.out_batch_stride        = convContext.out_channel_stride * convContext.n_outputs;
    convContext.has_active              = param.activation_param.has_active ? 1 : 0;
    convContext.rmv                     = rocm_meta_version::AMDHSA_1_0;
    convContext.use_binaries            = true;
    convContext.in_layout               = "NCHW";
    convContext.general_compile_options = " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";

    miopen::Db db = anakin::saber::GetDb(dev._info._device_name, dev._info._compute_core_num);
    miopen::Handle::setClEnv(context, device);
    miopen::Handle handle;
    convContext.SetStream(&handle);
    miopen::solver::ConvSolution solution = miopen::solver::SearchForSolution<
            miopen::solver::ConvBinWinograd3x3U,
            miopen::solver::ConvAsm3x3U,
            miopen::solver::ConvAsm1x1U,
            miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
            miopen::solver::ConvOclDirectFwdGen,
            miopen::solver::ConvOclDirectFwd3x3,
            miopen::solver::ConvOclDirectFwd1x1,
            miopen::solver::ConvOclDirectFwd>(convContext, db);
    miopen::Handle::clearClEnv();

    if (solution.workspce_sz > 0) {
        std::vector<Tensor<AMD>*> conv_outputs;
        Tensor<AMD>* conv_out = new Tensor<AMD>();
        conv_outputs.push_back(conv_out);
        Conv<AMD, AK_FLOAT> conv;
        conv.compute_output_shape(inputs, conv_outputs, param);
        conv_out->re_alloc(conv_out->shape());
        _outWorkspace = conv_out;
    }

    ALOGD("Solution wk size:" << solution.workspce_sz
                              << " param size:" << solution.construction_params.size());
    if (solution.construction_params.size() > 0) {
        for (auto miKernelInfo : solution.construction_params) {
            kernelInfo = miKernelInfo;

            kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);
            if (!kptr.get()->isInit()) {
                ALOGE("Failed to create kernel");
                return SaberInvalidValue;
            }
            _kernels_ptr.push_back(kptr);
        }
    } else {
        ALOGE("No solution found!!!");
    }

    ALOGD("COMPLETE CREATE KERNEL");

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberConv2D<AMD, OpDtype>::dispatch(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        ConvParam<AMD>& param) {

    ALOGD("dispatch");
    int err;
    amd_kernel_list list;
    bool isBias   = false;
    bool isActive = false;

    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    // To set the argument
    PtrDtype memObjects[4] = {0, 0, 0, 0};

    ALOGD(" num=" << inputs[0]->num() << " channel=" << inputs[0]->channel()
                  << " height=" << inputs[0]->height() << " width=" << inputs[0]->width()
                  << " param.weight()->num()=" << param.weight()->num()
                  << " param.weight()->channel()=" << param.weight()->channel()
                  << " param.weight()->width()=" << param.weight()->width()
                  << " param.weight()->height()=" << param.weight()->height() << " param.group="
                  << param.group << " param.pad_h=" << param.pad_h << " param.pad_w=" << param.pad_w
                  << " param.stride_h=" << param.stride_h << " param.stride_w=" << param.stride_w
                  << " param.dilation_h=" << param.dilation_h
                  << " param.dilation_w=" << param.dilation_w << " param.alpha=" << param.alpha
                  << " param.beta=" << param.beta);

    if (param.bias()->size() > 0) {
        isBias = true;
        ALOGD(" param.bias()->size()=" << param.bias()->size()
                                       << " param.bias()->channel()=" << param.bias()->channel()
                                       << " param.bias()->width()=" << param.bias()->width()
                                       << " param.bias()->height()=" << param.bias()->height());
    }
    if (param.activation_param.has_active) {
        isActive = true;
        ALOGD(" param.activation_param.has_active="
              << param.activation_param.has_active
              << " param.activation_param.negative_slope=" << param.activation_param.negative_slope
              << " param.activation_param.active=" << param.activation_param.active
              << " param.activation_param.coef=" << param.activation_param.coef);
    }

    if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
        ALOGE("Kernel is not exist");
        return SaberInvalidValue;
    }
    ALOGD("kernel size:" << _kernels_ptr.size() << " name:" << _kernels_ptr[0].get()->GetName());

    for (int i = 0; i < _kernels_ptr.size(); i++) {
        if ((_kernels_ptr[i].get()->GetName() == "MIOpenConvUni")
            || (_kernels_ptr[i].get()->GetName() == "MIOpenConv1x1")
            || (_kernels_ptr[i].get()->GetName() == "MIOpenConv1x1pquv")
            || (_kernels_ptr[i].get()->GetName() == "MIOpenCvD3x3_WSR0"))
        {
            memObjects[0] = (PtrDtype)inputs[0]->data();
            memObjects[1] = (PtrDtype)param.weight()->data();
            memObjects[2] = (isBias) ? (PtrDtype)param.bias()->data() : nullptr;
            memObjects[3] = (PtrDtype)outputs[0]->mutable_data();

            if (isBias) {
                if (isActive) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                            (PtrDtype)memObjects[0],
                            (PtrDtype)memObjects[1],
                            (PtrDtype)memObjects[2],
                            (PtrDtype)memObjects[3],
                            param.activation_param.negative_slope,
                            0.0f);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                            (PtrDtype)memObjects[0],
                            (PtrDtype)memObjects[1],
                            (PtrDtype)memObjects[2],
                            (PtrDtype)memObjects[3],
                            0.0f);
                }
            } else {
                if (isActive) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                            (PtrDtype)memObjects[0],
                            (PtrDtype)memObjects[1],
                            (PtrDtype)memObjects[3],
                            param.activation_param.negative_slope,
                            0.0f);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                            (PtrDtype)memObjects[0],
                            (PtrDtype)memObjects[1],
                            (PtrDtype)memObjects[3],
                            0.0f);
                }
            }
            if (!err) {
                ALOGE("Fail to set kernel args :" << err);
                return SaberInvalidValue;
            }
            list.push_back(_kernels_ptr[i]);
        }
        else if (_kernels_ptr[i].get()->GetName() == "sp3AsmConv3x3F")
        {
            int d_n_groups = 64, d_flags = 0;
            memObjects[0] = (PtrDtype)inputs[0]->data();
            memObjects[1] = (PtrDtype)param.weight()->data();
            memObjects[2] = (isBias) ? (PtrDtype)param.bias()->data() : 0;
            memObjects[3] = (PtrDtype)outputs[0]->mutable_data();

            err = _kernels_ptr[i].get()->SetKernelArgs(
                    (unsigned int)inputs[0]->num(),
                    (unsigned int)inputs[0]->channel(),
                    (unsigned int)inputs[0]->height(),
                    (unsigned int)inputs[0]->width(),
                    (unsigned int)param.weight()->num(),
                    (unsigned int)d_n_groups,
                    (unsigned int)d_flags,
                    param.activation_param.negative_slope,
                    (PtrDtype)memObjects[0],
                    (PtrDtype)memObjects[1],
                    (PtrDtype)memObjects[3],
                    (PtrDtype)memObjects[2]);

            if (!err) {
                ALOGE("Fail to set kernel args :" << err);
                return SaberInvalidValue;
            }
            list.push_back(_kernels_ptr[i]);
        }
        else
        {
            ALOGD("Not implement kernel name:" << _kernels_ptr[i].get()->GetName());
        }
    }
    err = LaunchKernel(cm, list);
    if (!err) {
        ALOGE("Fail to set execution :" << err);
        return SaberInvalidValue;
    }

    ALOGD("COMPLETE EXECUTION");

    return SaberSuccess;
}
template class SaberConv2D<AMD, AK_FLOAT>;
template class SaberConv2D<AMD, AK_HALF>;
template class SaberConv2D<AMD, AK_INT8>;
} // namespace saber
} // namespace anakin
