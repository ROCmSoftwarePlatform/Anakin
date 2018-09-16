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
#include "saber/funcs/impl/amd/include/amd_utils.h"

namespace anakin {
namespace saber {
// so that MIOpen works whether or not recent MIOpenGEMM changes pulled:
// // convert size_t and ulong kernel function parameters to unsigned.
namespace tempfix {
void set_offsets_to_uint(std::string& clstr, int times) {
    for (int i = 0; i < times; i++) {
        clstr = clstr.replace(clstr.find("const ulong"), 11, "const uint");
    }
}
void set_offsets_to_uint(std::string& clstr) {
    auto get_target = [](std::string inttype, char x) {
        std::stringstream ss;
        ss << "const " << inttype << ' ' << std::string(1, x) << "_offset";
        return std::regex(ss.str());
    };

    for (char x : {'a', 'b', 'c'}) {
        std::string replacement = "const unsigned " + std::string(1, x) + "_offset";
        for (auto inttype : {"size_t", "ulong"}) {
            clstr = std::regex_replace(clstr, get_target(inttype, x), replacement);
        }
    }
}
} // namespace tempfix
} // namespace saber
} // namespace anakin
