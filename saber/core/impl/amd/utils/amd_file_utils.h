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
#ifndef ANAKIN_SABER_FUNCS_IMPL_UTILS_AMDFILEUTILS_H
#define ANAKIN_SABER_FUNCS_IMPL_UTILS_AMDFILEUTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>
#include <openssl/md5.h>

namespace anakin {
namespace saber {

boost::filesystem::path
GetCacheFile(const std::string& device, const std::string& name, const std::string& args);

boost::filesystem::path GetCachePath();

void SaveBinary(
        const boost::filesystem::path& binary_path,
        const std::string& device,
        const std::string& name,
        const std::string& args);

std::string
LoadBinaryPath(const std::string& device, const std::string& name, const std::string& args);

std::string LoadFile(const std::string& s);

} // namespace saber
} // namespace anakin
#endif
