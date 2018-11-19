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
#ifndef ANAKIN_SABER_FUNC_IMPL_AMD_GEMM_H
#define ANAKIN_SABER_FUNC_IMPL_AMD_GEMM_H

#include <CL/cl.h>
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"
#include "saber/funcs/base.h"
#include <miopengemm/miogemm.hpp>
#include <miopengemm/gemm.hpp>
#include <miopengemm/geometry.hpp>

#define MLO_POOLING_OP_AVE 0
#define MLO_POOLING_OP_MAX 1

namespace anakin {
namespace saber {
typedef AMD_API::TPtr PtrDtype;

bool findGenericGemm(bool solver, std::vector<AMDKernelPtr>& kptr,
                     const std::vector<Tensor<AMD>*>& inputs,
                     std::vector<Tensor<AMD>*>& outputs,
                     ConvParam<AMD>& param,
                     PoolingParam<AMD>& pool_param,
                     Tensor<AMD>*& workspace,
                     Context<AMD>& ctx);

} // namespace saber
} // namespace anakin
#endif
