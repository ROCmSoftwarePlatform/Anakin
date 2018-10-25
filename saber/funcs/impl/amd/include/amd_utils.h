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
#ifndef ANAKIN_SABER_FUNC_IMPL_AMD_UTILS_H
#define ANAKIN_SABER_FUNC_IMPL_AMD_UTILS_H

#include <CL/cl.h>
#include "saber/core/impl/amd/utils/amd_base.h"

#define MLO_POOLING_OP_AVE 0
#define MLO_POOLING_OP_MAX 1

namespace anakin {
namespace saber {
// so that MIOpen works whether or not recent MIOpenGEMM changes pulled:
// convert size_t and ulong kernel function parameters to unsigned.
namespace tempfix {
void add_bias_relu(std::string& clstr);
void add_relu(std::string& clstr);
void set_offsets_to_uint(std::string& clstr, int times);
void set_offsets_to_uint(std::string& clstr);
} // namespace tempfix
} // namespace saber
} // namespace anakin
#endif
