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
#include "amd_file_utils.h"
#include <miopen/db_path.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "amd_logger.h"

namespace anakin {
namespace saber {

#define SABER_CACHE_DIR "~/.cache/amd_saber/"

std::string md5(std::string s) {
    std::array<unsigned char, MD5_DIGEST_LENGTH> result{};
    MD5(reinterpret_cast<const unsigned char*>(s.data()), s.length(), result.data());

    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (auto c : result)
        sout << std::setw(2) << int{c};

    return sout.str();
}

inline std::string
ReplaceString(std::string subject, const std::string& search, const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
    return subject;
}

boost::filesystem::path ComputeCachePath() {
    std::string cache_dir = SABER_CACHE_DIR;

    auto p = boost::filesystem::path{ReplaceString(cache_dir, "~", getenv("HOME"))};
    if (!boost::filesystem::exists(p))
        boost::filesystem::create_directories(p);
    AMD_LOGD("Get program path: " << p);
    return p;
}

boost::filesystem::path GetCachePath() {
    static const boost::filesystem::path path = ComputeCachePath();
    return path;
}

boost::filesystem::path
GetCacheFile(const std::string& device, const std::string& name, const std::string& args) {
    std::string filename = name + ".o";
    return GetCachePath() / md5(device + ":" + args) / filename;
}

std::string
LoadBinaryPath(const std::string& device, const std::string& name, const std::string& args) {
    auto f = GetCacheFile(device, name, args);
    if (boost::filesystem::exists(f)) {
        return f.string();
    } else {
        return {};
    }
}
void SaveBinary(
        const boost::filesystem::path& binary_path,
        const std::string& device,
        const std::string& name,
        const std::string& args) {
    auto p = GetCacheFile(device, name, args);
    boost::filesystem::create_directories(p.parent_path());
    boost::filesystem::rename(binary_path, p);
}

std::string LoadFile(const std::string& s) {
    std::ifstream t(s);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

miopen::Db GetDb(std::string device_name, int max_CU)
{
    auto p = boost::filesystem::path{ReplaceString(miopen::GetDbPath(), "~", getenv("HOME"))};
    if(!boost::filesystem::exists(p))
        boost::filesystem::create_directories(p);
    std::string dbFileName =
        p.string() + "/" + device_name + "_" + std::to_string(max_CU) + ".cd.pad.txt";
    return {dbFileName};
}
} // namespace saber
} // namespace anakin
