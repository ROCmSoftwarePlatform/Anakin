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

#include "miopen/gemm_utils.h"

namespace miopen {
namespace tempfix {
void set_offsets_to_uint(std::string& clstr, int times) {
    for (int i = 0; i < times; i++) {
        clstr = clstr.replace(clstr.find("const ulong"), 11, "const uint");
    }
}
} // namespace tempfix

#define WG_SIZE 256
#define MAX_ACTIVE_THREADS (64 * 4 * 64)

KernelInvoke transpose_NCHW2CNHW(Handle& handle,
                          int n,
                          int c,
                          int h_in,
                          int w_in,
                          int h_out,
                          int w_out,
                          int in_offset,
                          int out_offset,
                          int h_stride,
                          int w_stride)
{

    std::string program_name = "MIOpenUtilKernels4.cl";

    if(h_stride == 1 && w_stride == 1)
    {
        assert(h_in == h_out && w_in == w_out);

        std::string kernel_name = "transpose_NCHW2CNHW_opt";

        int RD_BLCK      = ((h_in * w_in) % 4 == 0) ? 4 : ((h_in * w_in) % 2 == 0) ? 2 : 1;
        int HW_RD        = (h_in * w_in) / RD_BLCK;
        size_t MAP_RD    = HW_RD * c;
        size_t lcl_size0 = WG_SIZE; //((MAP_RD + 63)/64 < 4) ? ((MAP_RD + 63)/64)*64 : 256;

        std::string READ_TYPE = (RD_BLCK == 1) ? "float" : "float" + std::to_string(RD_BLCK);

        std::string params;
        params += " -DNC_TRANS_NCHW_OPT";
        params += " -DIN_OFF=" + std::to_string(in_offset);
        params += " -DOUT_OFF=" + std::to_string(out_offset);
        params += " -DH=" + std::to_string(h_in);
        params += " -DW=" + std::to_string(w_in);
        params += " -DN=" + std::to_string(n);
        params += " -DC=" + std::to_string(c);
        params += " -DRD_BLCK=" + std::to_string(RD_BLCK);
        params += " -DHW_RD=" + std::to_string(HW_RD);
        params += " -DMAP_RD=" + std::to_string(MAP_RD);
        params += " -DREAD_TYPE=" + READ_TYPE;

        const std::vector<size_t> vld{lcl_size0, 1, 1};
        std::vector<size_t> vgd{MAP_RD, 1, 1};

        if(MAP_RD < MAX_ACTIVE_THREADS)
        {
            vgd = {MAP_RD, static_cast<size_t>(n), 1};
            params += " -D_2D_WG";
        }

        auto k = handle.AddKernel(
            kernel_name, "", program_name, kernel_name, vld, vgd, params);

        return k;
    }
    else
    {
        assert(h_in > h_out && w_in > w_out);

        std::string kernel_name = "transpose_NCHW2CNHW";

        std::string params;
        params += " -DNC_TRANS_NCHW";
        params += " -DN=" + std::to_string(n);
        params += " -DC=" + std::to_string(c);
        params += " -DHW_IN=" + std::to_string(h_in * w_in);
        params += " -DHW_OUT=" + std::to_string(h_out * w_out);
        params += " -DW_IN=" + std::to_string(w_in);
        params += " -DW_OUT=" + std::to_string(w_out);
        params += " -DH_STRIDE=" + std::to_string(h_stride);
        params += " -DW_STRIDE=" + std::to_string(w_stride);
        params += " -DIN_OFF=" + std::to_string(in_offset);
        params += " -DOUT_OFF=" + std::to_string(out_offset);

        size_t ld0 = WG_SIZE;
        size_t gd0 = c * h_out * w_out;
        const std::vector<size_t> vld{ld0, 1, 1};
        std::vector<size_t> vgd{gd0, 1, 1};

        if(gd0 < MAX_ACTIVE_THREADS)
        {
            vgd = {gd0, static_cast<size_t>(n), 1};
            params += " -D_2D_WG";
        }

        auto k = handle.AddKernel(
            kernel_name, "", program_name, kernel_name, vld, vgd, params);
        return k;
    }
}

KernelInvoke transpose_CNHW2NCHW(Handle& handle,
                          int n,
                          int c,
                          int h_out,
                          int w_out,
                          int h_in,
                          int w_in,
                          int in_offset,
                          int out_offset,
                          int h_stride,
                          int w_stride,
                          bool isBias)
{

    std::string program_name = "MIOpenUtilKernels4.cl";

    if(h_stride == 1 && w_stride == 1)
    {
        assert(h_out == h_in && w_out == w_in);

        std::string kernel_name = "transpose_CNHW2NCHW_opt_bias_prelu";

        int RD_BLCK      = ((h_out * w_out) % 4 == 0) ? 4 : ((h_out * w_out) % 2 == 0) ? 2 : 1;
        int HW_RD        = (h_out * w_out) / RD_BLCK;
        size_t MAP_RD    = HW_RD * c;
        size_t lcl_size0 = WG_SIZE; //((MAP_RD + 63)/64 < 4) ? ((MAP_RD + 63)/64)*64 : 256;

        std::string READ_TYPE = (RD_BLCK == 1) ? "float" : "float" + std::to_string(RD_BLCK);

        std::string params;
        params += " -DNC_TRANS_CNHW_OPT";
        params += " -DIN_OFF=" + std::to_string(in_offset);
        params += " -DOUT_OFF=" + std::to_string(out_offset);
        params += " -DH=" + std::to_string(h_out);
        params += " -DW=" + std::to_string(w_out);
        params += " -DN=" + std::to_string(n);
        params += " -DC=" + std::to_string(c);
        params += " -DRD_BLCK=" + std::to_string(RD_BLCK);
        params += " -DHW_RD=" + std::to_string(HW_RD);
        params += " -DMAP_RD=" + std::to_string(MAP_RD);
        params += " -DREAD_TYPE=" + READ_TYPE;

        if (isBias) {
            params += " -DBIAS";
        }

        const std::vector<size_t> vld{lcl_size0, 1, 1};
        std::vector<size_t> vgd{MAP_RD, 1, 1};

        if(MAP_RD < MAX_ACTIVE_THREADS)
        {
            vgd = {MAP_RD, static_cast<size_t>(n), 1};
            params += " -D_2D_WG";
        }

        auto k = handle.AddKernel(
            kernel_name, "", program_name, kernel_name, vld, vgd, params);

        return k;
    }
    else
    {
        assert(h_in > h_out && w_in > w_out);

        std::string kernel_name = "transpose_CNHW2NCHW";

        std::string params;
        params += " -DNC_TRANS_CNHW";
        params += " -DN=" + std::to_string(n);
        params += " -DC=" + std::to_string(c);
        params += " -DHW_IN=" + std::to_string(h_in * w_in);
        params += " -DHW_OUT=" + std::to_string(h_out * w_out);
        params += " -DW_IN=" + std::to_string(w_in);
        params += " -DW_OUT=" + std::to_string(w_out);
        params += " -DH_STRIDE=" + std::to_string(h_stride);
        params += " -DW_STRIDE=" + std::to_string(w_stride);
        params += " -DIN_OFF=" + std::to_string(in_offset);
        params += " -DOUT_OFF=" + std::to_string(out_offset);

        size_t ld0 = WG_SIZE;
        size_t gd0 = c * h_out * w_out;
        const std::vector<size_t> vld{ld0, 1, 1};
        std::vector<size_t> vgd{gd0, 1, 1};

        if(gd0 < MAX_ACTIVE_THREADS)
        {
            vgd = {gd0, static_cast<size_t>(n), 1};
            params += " -D_2D_WG";
        }

        auto k = handle.AddKernel(
            kernel_name, "", program_name, kernel_name, vld, vgd, params);

        return k;
    }
}
} // namespace miopen
