/********************************************************************************
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
 ********************************************************************************/

#ifndef MLO_INTERNAL_H_
#define MLO_INTERNAL_H_

// Header Files
#ifndef NOMINMAX
#define NOMINMAX // stupid windows.h confused with min() macros in std namespace
#endif

//#include <miopen/config.h>

#if 1
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#ifdef __APPLE__
#include <mach/mach_time.h> // for mach_absolute_time() and friends
#endif

#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <map>
#include <string>
#include <limits>
#include <algorithm> // std::find  and std::min std::maxx

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <ctime>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>
#include <cstdint>
#include <tuple>

using mlo_kernel_info = std::tuple<const std::string,
                                   const std::string,
                                   const std::string,
                                   const std::vector<size_t>,
                                   const std::vector<size_t>>;

//#if MIOPEN_BACKEND_OPENCL
//#include <miopen/oclkernel.hpp>
//#include <miopen/clhelper.hpp>
//#include <miopen/ocldeviceinfo.hpp>
//#endif
//#include <miopen/tensor.hpp>
#include <miopen/handle.hpp>
//#include <miopen/db_path.hpp>
//#include <miopen/db.hpp>

#define MLO_POOLING_OP_AVE 0
#define MLO_POOLING_OP_MAX 1

inline int mloLg2(int v)
{
    auto ret = static_cast<int>(std::ceil(std::log(v) / std::log(2)));
    return (ret);
}

inline int AlignUp(int val, unsigned step)
{
    assert(step > 0);
    return ((val + step - 1) / step) * step;
}

enum class rocm_meta_version
{
    Unknown,
    V1,
    V2,
    V3,
    AMDHSA_1_0,   // 1.0, see https://llvm.org/docs/AMDGPUUsage.html#code-object-metadata
    Default = V3, // Assumption for HIP backend. To be updated together with ROCm release.
};


namespace miopen {

template <class TInstance>
class StaticContainer
{
    public:
    inline static TInstance& Instance()
    {
        static TInstance data{};
        return data;
    }
};

struct ProblemDescription
{
    int n_inputs         = 0;
    int in_height        = 0;
    int in_width         = 0;
    int kernel_size1     = 0;
    int kernel_size0     = 0;
    int n_outputs        = 0;
    int out_height       = 0;
    int out_width        = 0;
    int batch_sz         = 0;
    int pad0             = 0;
    int pad1             = 0;
    int kernel_stride0   = 0;
    int kernel_stride1   = 0;
    int kernel_dilation0 = 0;
    int kernel_dilation1 = 0;
    int bias             = 0;
    std::string in_layout;
    std::string in_data_type{"FP32"};
    std::string weights_layout;
    std::string out_data_type;
    std::string out_layout;
    int float_size         = 32;
    size_t bot_sz          = 0;
    size_t top_sz          = 0;
    size_t weights_sz      = 0;
    size_t bias_sz         = 0;
    int deconvolution      = 0;
    int in_stride          = 0;
    int out_stride         = 0;
    int in_channel_stride  = 0;
    int in_batch_stride    = 0;
    int out_channel_stride = 0;
    int out_batch_stride   = 0;
    int group_counts       = 0;
    struct Direction
    {
        enum class Value
        {
            Unknown,
            Forward,
            Backward,
            BackwardWrW,
        };

        private:
        Value v = Value::Unknown;

        public:
        bool IsKnown() const { return v != Value::Unknown; }
        bool IsForward() const { return v == Value::Forward; }
        bool IsBackwardData() const { return v == Value::Backward; } // Syntax glue.
        bool IsBackwardWrW() const { return v == Value::BackwardWrW; }
        void Set(int forward)
        {
            assert(0 <= forward && forward <= 1);
            v = forward != 0 ? Value::Forward : Value::Backward;
        }
        template <typename T>
        void Set(T) = delete;
        void SetBackwardWrW() { v = Value::BackwardWrW; }
    } direction;
    int GetBackwardPad0() const { return kernel_size0 - pad0 - 1; }
    int GetBackwardPad1() const { return kernel_size1 - pad1 - 1; }

    void Serialize(std::ostream& stream) const
    {
        if(!direction.IsKnown())
            MIOPEN_THROW("!direction.IsKnown()");
        const auto sep = '-';
        // clang-format off
        // 576-4-4-1x1-192-4-4-8-1x1-2x2-3x3-0-NCHW-FP32-F
        stream
            << n_inputs << sep << in_height << sep << in_width
            << sep << kernel_size1 << 'x' << kernel_size0
            << sep << n_outputs << sep << out_height << sep << out_width
            << sep << batch_sz
            << sep << pad1 << 'x' << pad0
            << sep << kernel_stride1 << 'x' << kernel_stride0
            << sep << kernel_dilation1 << 'x' << kernel_dilation1
            << sep << bias
            << sep << in_layout
            << sep << in_data_type
            << sep << (direction.IsForward() ? "F"
                     : direction.IsBackwardData() ? "B" : "W"); // clang-format on
    }

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }
};

struct PoolingContext
{
    int batch_sz;
    int n_inputs;
    int in_height;
    int in_width;
    int n_outputs;
    int out_height;
    int out_width;
    int pooling_type;
    int pad1;
    int pad0;
    int kernel_size1;
    int kernel_size0;
    int kernel_stride1;
    int kernel_stride0;
};

/// A leftover of the legacy design, houses problem config,
/// environmental context (e.g. HW/SW platform) and solver-specific state.
///
/// TODO: These three entities should be made separate.
struct ConvolutionContext : ProblemDescription
{
    // Solution-specific
    std::string general_compile_options;
    // Operation modes & environment
    bool do_search                         = false;
    bool do_all_search                     = false;
    bool save_srch_req                     = false;
    bool use_asm_kernels                   = false;
    bool use_binaries                      = true;
    rocm_meta_version rmv                  = rocm_meta_version::Default;
    bool workaround_disable_search_enforce = false;
    bool has_active                        = false;
    float negative_slope                   = 0.0f;
    bool has_pooling                       = false;
    PoolingContext poolingContext;

    inline Handle& GetStream() const { return *_stream; }
    inline void SetStream(Handle* stream) { _stream = stream; }

    /*std::string GetPerfDbPath() const
    {
        // clang-format off
        return GetDbPath()
             + "/"
             + GetStream().GetDeviceName()
             + "_"
             + std::to_string(GetStream().GetMaxComputeUnits())
             + ".cd.pdb.txt";
        // clang-format on
    }

    std::string GetUserPerfDbPath() const
    {
        // clang-format off
        return GetUserDbPath()
             + "/"
             + GetStream().GetDeviceName()
             + "_"
             + std::to_string(GetStream().GetMaxComputeUnits())
             + ".cd.updb.txt";
        // clang-format on
    }*/

    private:
    Handle* _stream = nullptr;
};

/// Information required to build and run a kernel (or a set of kernels),
/// which is expected to perform computatons as per the problem config.
///
/// TODO: Currently best suits a subset of existing solvers,
/// namely some OpenCL-written forward direct convolutions.
/// Shall be refactored (possibly, to a class hierarchy).
/*struct ConvSolution
{
    std::vector<anakin::saber::KernelInfo> construction_params; // impl may consist of multiple kernels.
    miopenStatus_t status;
    int passes;

    size_t workspce_sz;
    int grp_tile1;
    int grp_tile0;
    int in_tile1;
    int in_tile0;
    int out_pix_tile1;
    int out_pix_tile0;
    int n_out_pix_tiles;
    int n_in_data_tiles;
    int n_stacks;

    ConvSolution(miopenStatus_t status_ = miopenStatusSuccess, int passes_ = 1)
        : status(status_),
          passes(passes_),
          workspce_sz(0),
          grp_tile1(-1),
          grp_tile0(-1),
          in_tile1(-1),
          in_tile0(-1),
          out_pix_tile1(-1),
          out_pix_tile0(-1),
          n_out_pix_tiles(-1),
          n_in_data_tiles(-1),
          n_stacks(-1)
    {
    }

    inline bool Succeeded() const { return status == miopenStatusSuccess; }
};*/

}
#endif
