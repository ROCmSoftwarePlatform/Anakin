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

#include "miopen/solver.hpp"
#include "miopen/solver_conv_common.hpp"
#include "miopen/boost_utils.hpp"
#include <TensileConv.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <sstream>
#include <vector>

namespace miopen {
namespace solver {

std::string changeTensileKernelName(int width, int height, int batch, int stride, int c, int k,
                                    int bias, int relu) {
    std::ostringstream newName;
    newName << TENSILE_KERNEL_DIR << "/ConvFwd1x1_" << width << "x" << height << "x" << c << "x" << k <<
            "x" << batch << "_" << bias << "_" << relu << "_" << stride << ".s";

    create_directories(TENSILE_KERNEL_DIR);

    std::ifstream in("./kernel/ConvFwd1x1.s", std::ios::in | std::ios::binary);
    std::ofstream out(newName.str(), std::ios::out | std::ios::binary);
    out << in.rdbuf();

    return newName.str();
}

std::string findTensileSolution(int width, int height, int batch, int stride, int channel, int k,
                                bool has_active,
                                int bias, std::vector<int>& local_worker_num, std::vector<int>& global_worker_num,
                                std::vector<int>& param_size, double& elapsed_time, bool auto_search) {

    TensileConv::DirConv1x1Fwd conv;
    TensileConv::T_TCSolution solution;
    TensileConv::E_TCRelu relu = has_active ? TensileConv::E_TCRelu::RELU :
                                 TensileConv::E_TCRelu::NORELU;

    TensileConv::E_TCSearch search = auto_search ? TensileConv::E_TCSearch::AUTO :
                                     TensileConv::E_TCSearch::NOSEARCH;

    elapsed_time = conv.TuneProblem(width, height, channel, k, batch, stride, stride, bias,
                                    relu, search, solution);

    std::cout << "kernel name:" << solution.kernel_name << std::endl;
    std::cout << "kernel file:" << solution.kernel_file << std::endl;
    std::cout << "group size:" << solution.GroupSize[0] << " " << solution.GroupSize[1] << " " <<
              solution.GroupSize[2] << std::endl;
    std::cout << "global size:" << solution.GlobalSize[0] << " " << solution.GlobalSize[1] << " " <<
              solution.GlobalSize[2] << std::endl;
    std::cout << "param size:" << solution.ParamSize[0] << " " << solution.ParamSize[1] << " " <<
              solution.ParamSize[2] << std::endl;
    std::cout << "min_proc_time:" << elapsed_time << std::endl;

    if (elapsed_time == -1) {
        return "";
    }

    elapsed_time *= 1000;

    local_worker_num[0]  = solution.GroupSize[0];
    global_worker_num[0] = solution.GlobalSize[0];

    local_worker_num[1]  = solution.GroupSize[1];
    global_worker_num[1] = solution.GlobalSize[1];

    local_worker_num[2]  = solution.GroupSize[2];
    global_worker_num[2] = solution.GlobalSize[2];

    param_size[0]        = solution.ParamSize[0];
    param_size[1]        = solution.ParamSize[1];
    param_size[2]        = solution.ParamSize[2];

    return changeTensileKernelName(width, height, batch, stride, channel, k, bias, has_active ? 1 : 0);
}

bool ConvOclDirectFwd1x1Tensile::IsApplicable(const ConvolutionContext& params) const {
    bool result =
        params.direction.IsForward() && (params.kernel_size0 == 1 && params.kernel_size1 == 1)
        && (params.kernel_stride0 <= 1)
        && (params.n_inputs % 4 == 0)
        && (params.n_outputs % 2 == 0)
        && (params.kernel_stride0 == params.kernel_stride1)
        && (params.pad0 == 0 && params.pad1 == 0);

    return result;
}

TensilePerformanceConfig ConvOclDirectFwd1x1Tensile::Search(const ConvolutionContext& params) {
    std::vector<int> localNum_v(3);
    std::vector<int> globalNum_v(3);
    std::vector<int> paramSize_v(3);
    double elapsed_time = 0;

    std::string kernel_file = findTensileSolution(params.in_width, params.in_height, params.batch_sz,
                              params.kernel_stride0, params.n_inputs, params.n_outputs,
                              params.has_active, params.bias, localNum_v, globalNum_v,
                              paramSize_v, elapsed_time, true);

    TensilePerformanceConfig config;

    config.kernel_file = kernel_file;
    config.local_worker_x = localNum_v[0];
    config.global_worker_x = globalNum_v[0];
    config.local_worker_y = localNum_v[1];
    config.global_worker_y = globalNum_v[1];
    config.local_worker_z = localNum_v[2];
    config.global_worker_z = globalNum_v[2];

    config.param_size_1 = paramSize_v[0];
    config.param_size_2 = paramSize_v[1];
    config.param_size_3 = paramSize_v[2];

    config.min_proc_time = elapsed_time;

    return config;
}

// dummy
TensilePerformanceConfig ConvOclDirectFwd1x1Tensile::GetPerformanceConfig(
    const ConvolutionContext& params) const {
    TensilePerformanceConfig config {};
    return config;
}

TensilePerformanceConfig
ConvOclDirectFwd1x1Tensile::SearchForMeasureOnce(const ConvolutionContext& params) const {
    TensilePerformanceConfig result;
    std::cout << "ConvOclDirectFwd1x1Tensile SearchForMeasureOnce" << std::endl;
    return result;
}

ConvSolution ConvOclDirectFwd1x1Tensile::GetSolution(
    const ConvolutionContext& params,
    const TensilePerformanceConfig& searched_params) const {
    ConvSolution result;

    std::vector<int> localNum_v(3);
    std::vector<int> globalNum_v(3);
    std::vector<int> paramSize_v(3);
    double elapsed_time = 0;

    std::string kernel_file = findTensileSolution(params.in_width, params.in_height, params.batch_sz,
                              params.kernel_stride0, params.n_inputs, params.n_outputs,
                              params.has_active, params.bias, localNum_v, globalNum_v,
                              paramSize_v, elapsed_time, false);
    KernelInfo kernelInfo;

    if (elapsed_time > 0) {
        kernelInfo.kernel_name = "ConvFwd1x1";
        kernelInfo.kernel_file = kernel_file;
        kernelInfo.l_wk = {localNum_v[0], localNum_v[1], localNum_v[2]};
        kernelInfo.g_wk = {globalNum_v[0], globalNum_v[1], globalNum_v[2]};

        kernelInfo.tensile_slot_size = paramSize_v[0];
        kernelInfo.tensile_l2_size = paramSize_v[1];
        kernelInfo.tensile_dbg_size = paramSize_v[2];

        kernelInfo.isMIOpenKernel = false;
        result.construction_params.push_back(kernelInfo);
    } else {
        result.status = miopenStatusInternalError;
        ALOGE("can NOT get solution");
    }

    result.min_proc_time = elapsed_time;
    return result;
}

} // namespace solver
} // namespace miopen
