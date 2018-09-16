#ifndef ANAKIN_SABER_FUNC_IMPL_AMD_UTILS_H
#define ANAKIN_SABER_FUNC_IMPL_AMD_UTILS_H

#include "saber/core/impl/amd/utils/amd_base.h"

#define MLO_POOLING_OP_MAX 0
#define MLO_POOLING_OP_AVE 1

namespace anakin {
namespace saber {
// so that MIOpen works whether or not recent MIOpenGEMM changes pulled:
// convert size_t and ulong kernel function parameters to unsigned.
namespace tempfix {
void set_offsets_to_uint(std::string& clstr, int times);
void set_offsets_to_uint(std::string& clstr);
} // namespace tempfix
} // namespace saber
} // namespace anakin
#endif
