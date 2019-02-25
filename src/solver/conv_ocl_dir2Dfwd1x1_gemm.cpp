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

//#include "miopen/solver.hpp"
#include "miopen/solver_conv_common.hpp"
#include "miopen/gemm_utils.h"
#include <miopengemm/miogemm.hpp>

namespace miopen {
namespace solver {

bool ConvOclDirectFwd1x1Gemm::IsApplicable(const ConvolutionContext& params) const {
    bool result =
        params.direction.IsForward() && (params.kernel_size0 == 1 && params.kernel_size1 == 1)
        && (params.batch_sz < 32) && (params.kernel_stride0 <= 2)
        && (params.kernel_stride0 == params.kernel_stride1)
        && (params.pad0 == 0 && params.pad1 == 0);

    return result;
}

ConvSolution ConvOclDirectFwd1x1Gemm::GetSolution(
    const ConvolutionContext& params,
    const LegacyPerformanceConfig& searched_params) const {
    ConvSolution result;
    result.min_proc_time = searched_params.min_proc_time; 

    KernelInfo kernelInfo;
    kernelInfo.kernel_name = "xGemm";
    result.construction_params.push_back(kernelInfo);

    return result;
}

template <class Solver>
static void ExpandGetSolution(ConvSolution& kernel_search_result, const ConvolutionContext& params, const LegacyPerformanceConfig& result, Solver&& s)
{
    if(!kernel_search_result.Succeeded()) // once
    {
        if(s.IsApplicable(params))
        {
            if(s.IsValidPerformanceConfig(params, result))
            {
                kernel_search_result = s.GetSolution(params, result);
            }
        }
    }
}

template <class Solver, class... Solvers>
static void ExpandGetSolution(ConvSolution& kernel_search_result, const ConvolutionContext& params, const LegacyPerformanceConfig& result, Solver&& solver, Solvers&&... solvers)
{
    ExpandGetSolution(kernel_search_result, result, solver);
    ExpandGetSolution(kernel_search_result, result, solvers...);
}

static void RunMiopengemmSolution(Handle* handle,
                           std::vector<KernelInvoke> kernels,
                           float alpha,
                           Data_t A,
                           int a_offset,
                           Data_t B,
                           int b_offset,
                           float beta,
                           Data_t C,
                           int c_offset,
                           double& processing_time) {

    double tmp_time = 0;
    const std::size_t kernel_size = kernels.size();

    if(kernel_size == 1)
    {
        // C = alpha * A * B + beta * C
        kernels[0](A, a_offset, B, b_offset, C, c_offset, alpha, beta);
        if(handle->IsProfilingEnabled())
            tmp_time = handle->GetKernelTime();
    }
    else if(kernel_size == 2)
    {
        // C *= beta
        kernels[1](C, c_offset, beta);

        if(handle->IsProfilingEnabled())
            tmp_time = handle->GetKernelTime();

        // C += alpha * A * B
        kernels[0](A, a_offset, B, b_offset, C, c_offset, alpha);

        if(handle->IsProfilingEnabled())
            tmp_time += handle->GetKernelTime();
    }
    else
    {
        MIOPEN_THROW("unable to get correct MIOpenGEMM kenerls");
    }
    std::cout << "tmp_time:" << tmp_time << std::endl;
    processing_time += tmp_time;
}

template <class... Solvers>
static int MeasureLoop(Handle* profile_h,
                       Data_t bot_ocl_buf,
                       Data_t top_ocl_buf,
                       Data_t wei_ocl_buf,
                       Data_t bias_ocl_buf,
                       double& processing_time,
                       const ConvolutionContext& params,
                       const LegacyPerformanceConfig& result)
{
    ConvSolution kernel_search_result{miopenStatusNotInitialized};

#if(__cplusplus >= 201402L)
    miopen::each_args(
        [&](auto s) {
            if(!kernel_search_result.Succeeded()) // once
            {
                if(s.IsApplicable(params))
                {
                    if(s.IsValidPerformanceConfig(params, result))
                    {
                        kernel_search_result = s.GetSolution(params, result);
                    }
                }
            }
        },
        Solvers{}...);
#else
    ExpandGetSolution(kernel_search_result, params, result, Solvers{}...);
#endif
    if(!kernel_search_result.Succeeded())
    {
        return 1;
    }

    MIOPEN_LOG_I2("Trying " << result);
    const auto kernel_params     = kernel_search_result.construction_params[0];
    std::string compiler_options = params.general_compile_options + kernel_params.comp_options;

    // Creating OCLKernel obj
    try
    {

        float padding_value = 0;

        if(profile_h)
        {
            processing_time = 0;//std::numeric_limits<float>::max();

            if ((params.batch_sz > 1 
                && (params.in_height <= 14 && params.in_height <= 14 && params.kernel_stride0 == 1))
                || params.kernel_stride0 == 2) {
                int K       = params.n_inputs;
                int M       = params.n_outputs;
                int N       = (params.batch_sz) * (params.out_height) * (params.out_width);
                float alpha = 1.0f;
                float beta  = 0.0f;
                bool transA     = false;
                bool transB     = false;
                bool transC     = false;
                int leadingd_A     = K;
                int leadingd_B     = N;
                int leadingd_C     = N;

                MIOpenGEMM::Geometry tgg {};
                tgg = MIOpenGEMM::Geometry(true, transB, transA, transC, leadingd_B, leadingd_A, leadingd_C, N, M,
                                           K, 0, 'f');

                size_t top_sz = params.batch_sz * 2 *
                                std::max({params.n_inputs, params.n_outputs}) * 
                                std::max({params.in_height, params.out_height}) * 
                                std::max({params.in_width, params.out_width});

                std::vector<float> tmp_top_sys_buf(top_sz);
                auto tmp_top_ocl_buf = profile_h->Write(tmp_top_sys_buf);

                profile_h->EnableProfiling();
                auto k = transpose_NCHW2CNHW(
                    *profile_h,
                    params.batch_sz,
                    params.n_inputs,
                    params.in_height,
                    params.in_width,
                    params.out_height,
                    params.out_width,
                    0,
                    0,
                    params.kernel_stride0,
                    params.kernel_stride1);

                k(bot_ocl_buf, tmp_top_ocl_buf.get());
                processing_time += profile_h->GetKernelTime();

                /////////////////////////////////////////////////////////////
                // gemm kernel
                // jn : print search results to terminal
                bool miopengemm_verbose = false;

                // jn : print warning messages when the returned kernel(s) might be sub-optimal
                bool miopengemm_warnings = false;

                MIOpenGEMM::Solution soln = MIOpenGEMM::find(0.003f,
                                                 profile_h->GetStream(),
                                                 tmp_top_ocl_buf.get(),
                                                 wei_ocl_buf,
                                                 tmp_top_ocl_buf.get(),
                                                 false,
                                                 tgg,
                                                 miopengemm_verbose,
                                                 miopengemm_warnings);

                std::string kernel_clstring = soln.v_tgks.back().kernstr;
                tempfix::set_offsets_to_uint(kernel_clstring, 3);

                std::string kernel_name = soln.v_tgks.back().fname;
                size_t local_work_size  = soln.v_tgks.back().local_work_size;
                size_t global_work_size = soln.v_tgks.back().global_work_size;

                std::vector<size_t> vld{local_work_size, 1, 1};
                std::vector<size_t> vgd{global_work_size, 1, 1};
                std::vector<KernelInvoke> kernels;

                k = profile_h->AddKernel("GEMM", "", kernel_clstring, kernel_name, vld, vgd, "", 0);
                kernels.push_back(k);

                if(soln.v_tgks.size() == 2)
                {
                    std::string beta_program_name = soln.v_tgks[0].kernstr;
                    tempfix::set_offsets_to_uint(beta_program_name, 1);

                    std::string beta_kernel_name = soln.v_tgks[0].fname;
                    local_work_size              = soln.v_tgks[0].local_work_size;
                    global_work_size             = soln.v_tgks[0].global_work_size;

                    vld[0] = local_work_size;
                    vgd[0] = global_work_size;

                    auto k2 = profile_h->AddKernel(
                        "GEMM", "", beta_program_name, beta_kernel_name, vld, vgd, "", 1);

                    kernels.push_back(k2);
                }
                
                RunMiopengemmSolution(profile_h,
                           kernels,
                           alpha,
                           tmp_top_ocl_buf.get(),
                           0,
                           wei_ocl_buf,
                           0,
                           beta,
                           tmp_top_ocl_buf.get(),
                           0,
                           processing_time); 

                size_t _x_t_size = params.batch_sz * params.n_inputs
                               * params.out_height * params.out_width; 

                k = transpose_CNHW2NCHW(
                    *profile_h,
                    params.batch_sz,
                    params.n_outputs,
                    params.out_height,
                    params.out_width,
                    params.in_height,
                    params.in_width,
                    _x_t_size,
                    0,
                    1,
                    1,
                    params.bias);

                if (params.bias) {
                    k(tmp_top_ocl_buf.get(), top_ocl_buf, bias_ocl_buf, params.negative_slope);
                    std::cout << "gemm bias" << std::endl;
                } else {
                    k(tmp_top_ocl_buf.get(), top_ocl_buf, params.negative_slope);
                    std::cout << "gemm non bias" << std::endl;
                }

                processing_time += profile_h->GetKernelTime();
                
            } else {
                int K = params.n_inputs;
                int M       = params.n_outputs;
                int N       = params.in_height * params.in_width;
                float alpha = 1.0;
                float beta  = 0.0;
                bool transA     = false;
                bool transB     = false;
                bool transC     = false;
                int leadingd_A     = K;
                int leadingd_B     = N;
                int leadingd_C     = N;

                profile_h->EnableProfiling();
                MIOpenGEMM::Geometry tgg {};
                tgg = MIOpenGEMM::Geometry(true, transB, transA, transC, leadingd_B, leadingd_A, leadingd_C, N, M,
                                           K, 0, 'f');

                /////////////////////////////////////////////////////////////
                // gemm kernel
                // jn : print search results to terminal
                bool miopengemm_verbose = false;

                // jn : print warning messages when the returned kernel(s) might be sub-optimal
                bool miopengemm_warnings = false;

                MIOpenGEMM::Solution soln = MIOpenGEMM::find(0.003f,
                                                 profile_h->GetStream(),
                                                 bot_ocl_buf,
                                                 wei_ocl_buf,
                                                 top_ocl_buf,
                                                 false,
                                                 tgg,
                                                 miopengemm_verbose,
                                                 miopengemm_warnings);

                std::string kernel_clstring = soln.v_tgks.back().kernstr;
                tempfix::set_offsets_to_uint(kernel_clstring, 3);

                std::string kernel_name = soln.v_tgks.back().fname;
                size_t local_work_size  = soln.v_tgks.back().local_work_size;
                size_t global_work_size = soln.v_tgks.back().global_work_size;

                std::vector<size_t> vld{local_work_size, 1, 1};
                std::vector<size_t> vgd{global_work_size, 1, 1};
                std::vector<KernelInvoke> kernels;

                auto k = profile_h->AddKernel("GEMM", "", kernel_clstring, kernel_name, vld, vgd, "", 0);
                kernels.push_back(k);

                if(soln.v_tgks.size() == 2)
                {
                    std::string beta_program_name = soln.v_tgks[0].kernstr;
                    tempfix::set_offsets_to_uint(beta_program_name, 1);

                    std::string beta_kernel_name = soln.v_tgks[0].fname;
                    local_work_size              = soln.v_tgks[0].local_work_size;
                    global_work_size             = soln.v_tgks[0].global_work_size;

                    vld[0] = local_work_size;
                    vgd[0] = global_work_size;

                    auto k2 = profile_h->AddKernel(
                        "GEMM", "", beta_program_name, beta_kernel_name, vld, vgd, "", 1);

                    kernels.push_back(k2);
                }

                unsigned int out_offset = 0;
                unsigned int in_offset  = 0;

                for (int j = 0; j < (params.batch_sz); j++) { 
                    in_offset = j * params.n_inputs * params.in_height
                            * params.in_width;
                    out_offset = j * params.n_outputs * params.out_height
                             * params.out_width;    

                    RunMiopengemmSolution(profile_h,
                           kernels,
                           alpha,
                           bot_ocl_buf,
                           in_offset,
                           wei_ocl_buf,
                           0,
                           beta,
                           top_ocl_buf,
                           out_offset,
                           processing_time);
                }
            }
        }
    }

    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_E("MeasureLoop failed for: " << ex.what());
        return -1;
    }

    //MIOPEN_LOG_I2("\t\t\t\t" << processing_time);
    std::cout << "\t\t\t\t" << processing_time << std::endl;
    return 0;
}
        

LegacyPerformanceConfig
ConvOclDirectFwd1x1Gemm::SearchForMeasureOnce(const ConvolutionContext& params) const
{
    LegacyPerformanceConfig result;

    miopen::Handle profile_h;
    double processing_time = std::numeric_limits<double>::max();
    double min_proc_time = std::numeric_limits<double>::max();

    // enable profiling for the handle for benchmarking
    //profile_h.EnableProfiling(true);

    // allocate tem input/output buffers
    size_t bot_sz = params.bot_sz / sizeof(float);
    std::vector<float> bot_sys_buf(bot_sz);

    for(int i = 0; i < bot_sz; i++)
    {
        bot_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
    }

    auto bot_ocl_buf = profile_h.Write(bot_sys_buf);

    size_t top_sz = params.top_sz / sizeof(float);
    std::vector<float> top_sys_buf(top_sz);

    auto top_ocl_buf = profile_h.Write(top_sys_buf);

    std::vector<float> random_top_sys_buf(top_sz);
    for(int i = 0; i < top_sz; i++)
    {
        random_top_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
    }

    size_t weights_sz = params.weights_sz / sizeof(float);
    std::vector<float> wei_sys_buf(weights_sz);

    std::cout << "top_sz:" << top_sz << " bot_sz:" << bot_sz << " weights_sz:" << weights_sz << std::endl;

    for(int i = 0; i < weights_sz; i++)
    {
        wei_sys_buf[i] = static_cast<float>((rand() * (1.0 / RAND_MAX) - 0.5) * 0.001);
    }

    auto wei_ocl_buf = profile_h.Write(wei_sys_buf);

    std::vector<float> bias_sys_buf;
    miopen::Allocator::ManageDataPtr bias_ocl_buf = nullptr;

    if(params.bias != 0)
    {
        size_t bias_sz = params.bias_sz / sizeof(float);
        bias_sys_buf   = std::vector<float>(bias_sz);
        for(int i = 0; i < bias_sz; i++)
        {
            bias_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
        }

        bias_ocl_buf = profile_h.Write(bias_sys_buf);
    }

    // randomize output
    profile_h.WriteTo(reinterpret_cast<const void*>(random_top_sys_buf.data()),
          top_ocl_buf,
          random_top_sys_buf.size() * sizeof(float));
    const auto ret = MeasureLoop<ConvOclDirectFwd1x1Gemm>(&profile_h,
                              bot_ocl_buf.get(),
                              top_ocl_buf.get(),
                              wei_ocl_buf.get(),
                              bias_ocl_buf.get(),
                              processing_time,
                              params,
                              result);
    
    if(min_proc_time > processing_time)
    {
        min_proc_time = processing_time;
        MIOPEN_LOG_I2("processing_time = " << processing_time << ", result = " << result);
    }

    std::cout << std::endl << "Score: " << min_proc_time << std::endl;

    result.grp_tile0       = 0;
    result.grp_tile1       = 0;
    result.in_tile0        = 0;
    result.in_tile1        = 0;
    result.out_pix_tile0   = 0;
    result.out_pix_tile1   = 0;
    result.n_out_pix_tiles = 0;
    result.n_in_data_tiles = 0;
    result.n_stacks        = 0;
    result.min_proc_time   = min_proc_time;

    profile_h.EnableProfiling(false);
    return result;
}

} // namespace solver
} // namespace miopen
