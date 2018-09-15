#include <limits>
#include "include/saber_softmax.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberSoftmax<AMD, OpDtype>::init(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        SoftmaxParam<AMD>& param,
        Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberSoftmax<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);
    if (!kptr.get()->isInit()) {
        ALOGE("Failed to load program");
        return SaberInvalidValue;
    }
    _kernels.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus SaberSoftmax<AMD, OpDtype>::create(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        SoftmaxParam<AMD>& param,
        Context<AMD>& ctx) {
    _kernels.clear();

    //! compute size
    Shape shape_in  = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    CHECK_EQ(shape_in == shape_out, true) << "valid shapes must be the same";
    _outer_num = inputs[0]->count_valid(0, param.axis);
    _inner_num = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
    _axis_size = shape_in[param.axis];

    size_t sharedmem_size = 32768;
    _max_dimsize          = sharedmem_size / sizeof(OpDtype) / AMD_NUM_THREADS;

    Shape sh_tmp({1, 1, 1, _outer_num * _inner_num});
    if (_axis_size > _max_dimsize) {
        //! re_alloc device memory
        _max_data.reshape(sh_tmp);
        _sum_data.reshape(sh_tmp);
    }

    //! CHECK whether the input or output tensor is with continuous buffer or not
    _is_continue_buf = outputs[0]->is_continue_mem() && inputs[0]->is_continue_mem();
    _dims            = shape_in.size();
    if (!_is_continue_buf) {
        Shape sh_input_real_stride  = inputs[0]->get_stride();
        Shape sh_output_real_stride = outputs[0]->get_stride();

        //! re_alloc device memory
        Shape sh({1, 1, 1, _dims});
        _valid_shape.reshape(sh);
        _input_stride.reshape(sh);
        _output_stride.reshape(sh);

        TargetWrapper<AMD, __device_target>::sync_memcpy(
                _valid_shape.mutable_data(),
                0,
                0,
                inputs[0]->valid_shape().data(),
                0,
                0,
                sizeof(int) * _dims,
                __HtoD());
        TargetWrapper<AMD, __device_target>::sync_memcpy(
                _input_stride.mutable_data(),
                0,
                0,
                sh_input_real_stride.data(),
                0,
                0,
                sizeof(int) * _dims,
                __HtoD());
        TargetWrapper<AMD, __device_target>::sync_memcpy(
                _output_stride.mutable_data(),
                0,
                0,
                sh_output_real_stride.data(),
                0,
                0,
                sizeof(int) * _dims,
                __HtoD());
    }

    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "Softmax.cl";
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {256};
    kernelInfo.g_wk = {(_inner_num * _outer_num + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]
                       * kernelInfo.l_wk[0]};

    if (_is_continue_buf) {
        //! softmax kernel without roi
        if (this->_axis_size <= _max_dimsize) {
            kernelInfo.kernel_name = "sharemem_softmax_kernel";
            CreateKernelList(inputs[0]->device_id(), kernelInfo);
        } else {
            //! firstly, get maximum data
            kernelInfo.kernel_name = "softmax_max_kernel";
            CreateKernelList(inputs[0]->device_id(), kernelInfo);
            //! then, compute exp and sum data
            kernelInfo.kernel_name = "softmax_sub_exp_sum_kernel";
            CreateKernelList(inputs[0]->device_id(), kernelInfo);
            //! lastly, compute divided output
            kernelInfo.kernel_name = "softmax_divid_output_kernel";
            CreateKernelList(inputs[0]->device_id(), kernelInfo);
        }
    } else {
        //! softmax kernel with roi
        if (this->_axis_size <= _max_dimsize) {
            kernelInfo.kernel_name = "sharemem_softmax_roi_kernel";
            CreateKernelList(inputs[0]->device_id(), kernelInfo);
        } else {
            //! firstly, get maximum data
            kernelInfo.kernel_name = "softmax_max_roi_kernel";
            CreateKernelList(inputs[0]->device_id(), kernelInfo);
            //! then, compute exp and sum data
            kernelInfo.kernel_name = "softmax_sub_exp_sum_roi_kernel";
            CreateKernelList(inputs[0]->device_id(), kernelInfo);
            //! lastly, compute divided output
            kernelInfo.kernel_name = "softmax_divid_output_roi_kernel";
            CreateKernelList(inputs[0]->device_id(), kernelInfo);
        }
    }
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSoftmax<AMD, OpDtype>::dispatch(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        SoftmaxParam<AMD>& param) {

    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    //! inputs only has one tensor
    int total_threads            = this->_inner_num * this->_outer_num;
    const OpDataType* data_in    = (const OpDataType*)inputs[0]->data();
    OpDataType* data_out         = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* max_data         = (OpDataType*)this->_max_data.mutable_data();
    OpDataType* sum_data         = (OpDataType*)this->_sum_data.mutable_data();
    const int* valid_shape       = (const int*)_valid_shape.data();
    const int* input_stride      = (const int*)_input_stride.data();
    const int* output_stride     = (const int*)_output_stride.data();
    bool err                     = false;
    amd_kernel_list::iterator it = _kernels.begin();

    if (_is_continue_buf) {
        //! softmax kernel without roi
        if (this->_axis_size <= _max_dimsize) {
            err = it->get()->SetKernelArgs(
                    (int)total_threads,
                    (PtrDtype)data_in,
                    (PtrDtype)data_out,
                    (int)this->_inner_num,
                    (int)this->_outer_num,
                    (int)this->_axis_size);
            if (!err) {
                LOG(ERROR) << "Fail to set execution";
                return SaberInvalidValue;
            }
        } else {
            //! firstly, get maximum data
            OpDataType min_data = std::numeric_limits<OpDataType>::min();
            err                 = it++->get()->SetKernelArgs(
                    (int)total_threads,
                    (PtrDtype)data_in,
                    (PtrDtype)max_data,
                    (float)min_data,
                    (int)this->_inner_num,
                    (int)this->_outer_num,
                    (int)this->_axis_size);
            if (!err) {
                LOG(ERROR) << "Fail to set execution";
                return SaberInvalidValue;
            }
            //! then, compute exp and sum data
            err = it++->get()->SetKernelArgs(
                    (int)total_threads,
                    (PtrDtype)data_in,
                    (PtrDtype)data_out,
                    (PtrDtype)max_data,
                    (PtrDtype)sum_data,
                    (int)this->_inner_num,
                    (int)this->_outer_num,
                    (int)this->_axis_size);
            if (!err) {
                LOG(ERROR) << "Fail to set execution";
                return SaberInvalidValue;
            }
            //! lastly, compute divided output
            err = it->get()->SetKernelArgs(
                    (int)total_threads,
                    (PtrDtype)data_out,
                    (PtrDtype)sum_data,
                    (int)this->_inner_num,
                    (int)this->_outer_num,
                    (int)this->_axis_size);
            if (!err) {
                LOG(ERROR) << "Fail to set execution";
                return SaberInvalidValue;
            }
        }
    } else {
        //! softmax kernel with roi
        if (this->_axis_size <= _max_dimsize) {
            err = it->get()->SetKernelArgs(
                    (int)total_threads,
                    (PtrDtype)data_in,
                    (PtrDtype)data_out,
                    (PtrDtype)input_stride,
                    (PtrDtype)output_stride,
                    (PtrDtype)valid_shape,
                    (int)param.axis,
                    (int)this->_axis_size,
                    (int)this->_dims);
            if (!err) {
                LOG(ERROR) << "Fail to set execution";
                return SaberInvalidValue;
            }
        } else {
            //! firstly, get maximum data
            OpDataType min_data = std::numeric_limits<OpDataType>::min();
            err                 = it++->get()->SetKernelArgs(
                    (int)total_threads,
                    (PtrDtype)data_in,
                    (PtrDtype)max_data,
                    (float)min_data,
                    (PtrDtype)input_stride,
                    (PtrDtype)output_stride,
                    (PtrDtype)valid_shape,
                    (int)param.axis,
                    (int)this->_axis_size,
                    (int)this->_dims);
            if (!err) {
                LOG(ERROR) << "Fail to set execution";
                return SaberInvalidValue;
            }
            //! then, compute exp and sum data
            err = it++->get()->SetKernelArgs(
                    (int)total_threads,
                    (PtrDtype)data_in,
                    (PtrDtype)data_out,
                    (PtrDtype)max_data,
                    (PtrDtype)sum_data,
                    (PtrDtype)input_stride,
                    (PtrDtype)output_stride,
                    (PtrDtype)valid_shape,
                    (int)param.axis,
                    (int)this->_axis_size,
                    (int)this->_dims);
            if (!err) {
                LOG(ERROR) << "Fail to set execution";
                return SaberInvalidValue;
            }
            //! lastly, compute divided output
            err = it->get()->SetKernelArgs(
                    (int)total_threads,
                    (PtrDtype)data_out,
                    (PtrDtype)sum_data,
                    (PtrDtype)input_stride,
                    (PtrDtype)output_stride,
                    (PtrDtype)valid_shape,
                    (int)param.axis,
                    (int)this->_axis_size,
                    (int)this->_dims);
            if (!err) {
                LOG(ERROR) << "Fail to set execution";
                return SaberInvalidValue;
            }
        }
    }

    err = LaunchKernel(cm, _kernels);

    if (!err) {
        ALOGE("Fail to set execution");
        return SaberInvalidValue;
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberSoftmax, SoftmaxParam, AMD, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSoftmax, SoftmaxParam, AMD, AK_INT8);
} // namespace saber

} // namespace anakin
