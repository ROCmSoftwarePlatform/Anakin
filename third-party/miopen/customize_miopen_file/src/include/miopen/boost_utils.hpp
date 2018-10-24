/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#ifndef ANAKIN_SABER_FUNCS_IMPL_UTILS_AMDFILEUTILS_H
#define ANAKIN_SABER_FUNCS_IMPL_UTILS_AMDFILEUTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <openssl/md5.h>
#include <miopen/db.hpp>

namespace miopen {

std::string temp_directory_path();
bool is_directory(const std::string& path);
std::string filename(const std::string& filename);
std::string remove_filename(std::string path);
int permissions(std::string& p, mode_t mode);
std::string unique_path();
bool exists(std::string path);
std::string parent_path(const std::string& path);
bool create_directories(std::string path);
std::string genTempFilePath(std::string name);
void writeFile(const std::string& content, const std::string& name);

} // namespace miopen
#endif
