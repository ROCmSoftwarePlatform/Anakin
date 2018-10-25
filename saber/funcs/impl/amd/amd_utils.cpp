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
// convert size_t and ulong kernel function parameters to unsigned.
namespace tempfix {
void add_bias_relu(std::string& clstr) {
    clstr = clstr.insert(
            clstr.find("miog_betac_alphaab") + 20,
            "__constant TFLOAT * restrict bias,\nTFLOAT slope,");

    std::string search      = "c[index] += alpha*rC";
    std::string sub_search1 = "c[index] += alpha*rC[dima][dimb]";
    std::string sub_search2 =
            "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb]";
    std::string sub_search3 = "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
                              "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string sub_search4 =
            "c[index] += alpha*rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string add1 = "rC[dima][dimb] += bias[write_start_b + dimb];\nrC[dima][dimb] *= "
                       "(rC[dima][dimb] > 0.0f ? 1.0f : slope);\n";
    std::string add2 =
            "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] += bias[write_start_b + "
            "dimb];\nrC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] *= "
            "(rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] > 0.0f ? 1.0f : slope);\n";
    std::string add3 =
            "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + "
            "dimbi_v] += bias[write_start_b + dimb];\nrC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
            "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] *= "
            "(rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + "
            "dimbi_v] > 0.0f ? 1.0f : slope);\n";
    std::string add4 =
            "rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] += bias[write_start_b + "
            "dimb];\nrC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] *= "
            "(rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] > 0.0f ? 1.0f : slope);\n";

    for (size_t pos = clstr.find(search); pos != std::string::npos; pos = clstr.find(search, pos)) {
        size_t temp = clstr.find(sub_search2);
        if (clstr.find(sub_search1) != std::string::npos) {
            clstr = clstr.insert(pos, add1);
            pos += add1.length() + sub_search1.length();
        } else if (clstr.find(sub_search2) != std::string::npos) {
            clstr = clstr.insert(pos, add2);
            pos += add2.length() + sub_search2.length();
        } else if (clstr.find(sub_search3) != std::string::npos) {
            clstr = clstr.insert(pos, add3);
            pos += add3.length() + sub_search3.length();
        } else if (clstr.find(sub_search4) != std::string::npos) {
            clstr = clstr.insert(pos, add4);
            pos += add4.length() + sub_search4.length();
        } else {
            break;
        }
    }
}

void add_relu(std::string& clstr) {
    clstr = clstr.insert(clstr.find("miog_betac_alphaab") + 20, "TFLOAT slope,");

    std::string search      = "c[index] += alpha*rC";
    std::string sub_search1 = "c[index] += alpha*rC[dima][dimb]";
    std::string sub_search2 =
            "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb]";
    std::string sub_search3 = "c[index] += alpha*rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
                              "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string sub_search4 =
            "c[index] += alpha*rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v]";
    std::string add1 = "rC[dima][dimb] *= (rC[dima][dimb] > 0.0f ? 1.0f : slope);\n";
    std::string add2 =
            "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] *= "
            "(rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][dimb] > 0.0f ? 1.0f : slope);\n";
    std::string add3 =
            "rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + "
            "dimbi_v] *= (rC[(dimai*VEW_A)/N_MICRO_IN_MACRO_A + "
            "dimai_v][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] > 0.0f ? 1.0f : slope);\n";
    std::string add4 =
            "rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] *= "
            "(rC[dima][(dimbi*VEW_B)/N_MICRO_IN_MACRO_B + dimbi_v] > 0.0f ? 1.0f : slope);\n";

    for (size_t pos = clstr.find(search); pos != std::string::npos; pos = clstr.find(search, pos)) {
        size_t temp = clstr.find(sub_search2);
        if (clstr.find(sub_search1) != std::string::npos) {
            clstr = clstr.insert(pos, add1);
            pos += add1.length() + sub_search1.length();
        } else if (clstr.find(sub_search2) != std::string::npos) {
            clstr = clstr.insert(pos, add2);
            pos += add2.length() + sub_search2.length();
        } else if (clstr.find(sub_search3) != std::string::npos) {
            clstr = clstr.insert(pos, add3);
            pos += add3.length() + sub_search3.length();
        } else if (clstr.find(sub_search4) != std::string::npos) {
            clstr = clstr.insert(pos, add4);
            pos += add4.length() + sub_search4.length();
        } else {
            break;
        }
    }
}

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

#define WG_SIZE 256
#define MAX_ACTIVE_THREADS (64 * 4 * 64)

void Im2ColGPU(
        KernelInfo& kernelInfo,
        AMDKernelPtr& kptr,
        int device_id,
        int c,
        int h,
        int w,
        int wei_h,
        int wei_w,
        int out_h,
        int out_w,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w) {
    kernelInfo.kernel_file = "MIOpenUtilKernels.cl";
    kernelInfo.kernel_name = "Im2Col";
    kernelInfo.kernel_type = MIOPEN;

    std::string params;
    int num_ch_per_wg;
    if ((out_h <= 8 && out_w <= 8) && (stride_h == 1 && stride_w == 1) && (c % 4 == 0))
        num_ch_per_wg = 4;
    else
        num_ch_per_wg = 1;

    int tile_sz_x  = 32;
    int tile_sz_y  = 8;
    int num_blks_x = std::ceil(static_cast<float>(out_w) / tile_sz_x);
    int num_blks   = num_blks_x * std::ceil(static_cast<float>(out_h) / tile_sz_y);
    int local_mem_sz;
    if (num_ch_per_wg == 1)
        local_mem_sz = ((tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1)
                       * ((tile_sz_y - 1) * stride_h + (wei_h - 1) * dilation_h + 1);
    else
        local_mem_sz = std::max(
                num_ch_per_wg
                        * ((std::ceil(static_cast<float>(tile_sz_x) / num_ch_per_wg) - 1) * stride_w
                           + (wei_w - 1) * dilation_w + 1)
                        * ((tile_sz_y - 1) * stride_h + (wei_h - 1) * dilation_h + 1),
                num_ch_per_wg * ((tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1)
                        * ((std::ceil(static_cast<float>(tile_sz_y) / num_ch_per_wg) - 1) * stride_h
                           + (wei_h - 1) * dilation_h + 1));

    // int data_size_off = data_size - im_offset;

    params += " -DNUM_CH_PER_WG=" + std::to_string(num_ch_per_wg);
    params += " -DNUM_IM_BLKS_X=" + std::to_string(num_blks_x);
    params += " -DNUM_IM_BLKS=" + std::to_string(num_blks);
    params += " -DLOCAL_MEM_SIZE=" + std::to_string(local_mem_sz);
    params += " -DSTRIDE_GT_1=" + std::to_string(static_cast<int>(stride_h * stride_w > 1));
    params += " -DTILE_SZ_X=" + std::to_string(tile_sz_x);
    params += " -DTILE_SZ_Y=" + std::to_string(tile_sz_y);
    params += " -DUSE_IM_OFF_GUARD=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1";

    kernelInfo.l_wk = {256, 1, 1};
    kernelInfo.g_wk = {256 * std::max(1, (c / num_ch_per_wg)) * num_blks, 1, 1};

    kernelInfo.comp_options = params;

    kptr = CreateKernel(device_id, &kernelInfo);
}

void transpose_NCHW2CNHW(
        KernelInfo& kernelInfo,
        AMDKernelPtr& kptr,
        int device_id,
        int n,
        int c,
        int h_in,
        int w_in,
        int h_out,
        int w_out,
        int in_offset,
        int out_offset,
        int h_stride,
        int w_stride) {

    kernelInfo.kernel_file = "MIOpenUtilKernels4.cl";

    if (h_stride == 1 && w_stride == 1) {
        kernelInfo.kernel_name = "transpose_NCHW2CNHW_opt";
        kernelInfo.kernel_type = MIOPEN;

        int RD_BLCK      = ((h_in * w_in) % 4 == 0) ? 4 : ((h_in * w_in) % 2 == 0) ? 2 : 1;
        int HW_RD        = (h_in * w_in) / RD_BLCK;
        size_t MAP_RD    = HW_RD * c;
        size_t lcl_size0 = WG_SIZE; //((MAP_RD + 63)/64 < 4) ? ((MAP_RD + 63)/64)*64 : 256;

        std::string READ_TYPE = (RD_BLCK == 1) ? "float" : "float" + std::to_string(RD_BLCK);

        kernelInfo.comp_options = "";
        kernelInfo.comp_options += " -DNC_TRANS_NCHW_OPT";
        kernelInfo.comp_options += " -DIN_OFF=" + std::to_string(in_offset);
        kernelInfo.comp_options += " -DOUT_OFF=" + std::to_string(out_offset);
        kernelInfo.comp_options += " -DH=" + std::to_string(h_in);
        kernelInfo.comp_options += " -DW=" + std::to_string(w_in);
        kernelInfo.comp_options += " -DN=" + std::to_string(n);
        kernelInfo.comp_options += " -DC=" + std::to_string(c);
        kernelInfo.comp_options += " -DRD_BLCK=" + std::to_string(RD_BLCK);
        kernelInfo.comp_options += " -DHW_RD=" + std::to_string(HW_RD);
        kernelInfo.comp_options += " -DMAP_RD=" + std::to_string(MAP_RD);
        kernelInfo.comp_options += " -DREAD_TYPE=" + READ_TYPE;

        kernelInfo.l_wk = {lcl_size0, 1, 1};
        kernelInfo.g_wk = {MAP_RD, 1, 1};

        if (MAP_RD < MAX_ACTIVE_THREADS) {
            kernelInfo.g_wk = {MAP_RD, static_cast<size_t>(n), 1};
            kernelInfo.comp_options += " -D_2D_WG";
        }

        kptr = CreateKernel(device_id, &kernelInfo);

    } else {
        kernelInfo.kernel_name = "transpose_NCHW2CNHW";
        kernelInfo.kernel_type = MIOPEN;

        kernelInfo.comp_options = "";
        kernelInfo.comp_options += " -DNC_TRANS_NCHW";
        kernelInfo.comp_options += " -DN=" + std::to_string(n);
        kernelInfo.comp_options += " -DC=" + std::to_string(c);
        kernelInfo.comp_options += " -DHW_IN=" + std::to_string(h_in * w_in);
        kernelInfo.comp_options += " -DHW_OUT=" + std::to_string(h_out * w_out);
        kernelInfo.comp_options += " -DW_IN=" + std::to_string(w_in);
        kernelInfo.comp_options += " -DW_OUT=" + std::to_string(w_out);
        kernelInfo.comp_options += " -DH_STRIDE=" + std::to_string(h_stride);
        kernelInfo.comp_options += " -DW_STRIDE=" + std::to_string(w_stride);
        kernelInfo.comp_options += " -DIN_OFF=" + std::to_string(in_offset);
        kernelInfo.comp_options += " -DOUT_OFF=" + std::to_string(out_offset);

        size_t ld0      = WG_SIZE;
        size_t gd0      = c * h_out * w_out;
        kernelInfo.l_wk = {ld0, 1, 1};
        kernelInfo.g_wk = {gd0, 1, 1};

        if (gd0 < MAX_ACTIVE_THREADS) {
            kernelInfo.g_wk = {gd0, static_cast<size_t>(n), 1};
            kernelInfo.comp_options += " -D_2D_WG";
        }

        kptr = CreateKernel(device_id, &kernelInfo);
    }
}

void transpose_CNHW2NCHW(
        KernelInfo& kernelInfo,
        AMDKernelPtr& kptr,
        int device_id,
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
        bool isBias) {

    kernelInfo.kernel_file = "MIOpenUtilKernels4.cl";

    if (h_stride == 1 && w_stride == 1) {
        kernelInfo.kernel_name = "transpose_CNHW2NCHW_opt_bias_prelu";
        kernelInfo.kernel_type = MIOPEN;

        int RD_BLCK      = ((h_out * w_out) % 4 == 0) ? 4 : ((h_out * w_out) % 2 == 0) ? 2 : 1;
        int HW_RD        = (h_out * w_out) / RD_BLCK;
        size_t MAP_RD    = HW_RD * c;
        size_t lcl_size0 = WG_SIZE; //((MAP_RD + 63)/64 < 4) ? ((MAP_RD + 63)/64)*64 : 256;

        std::string READ_TYPE = (RD_BLCK == 1) ? "float" : "float" + std::to_string(RD_BLCK);

        kernelInfo.comp_options = "";
        kernelInfo.comp_options += " -DNC_TRANS_CNHW_OPT";
        kernelInfo.comp_options += " -DIN_OFF=" + std::to_string(in_offset);
        kernelInfo.comp_options += " -DOUT_OFF=" + std::to_string(out_offset);
        kernelInfo.comp_options += " -DH=" + std::to_string(h_out);
        kernelInfo.comp_options += " -DW=" + std::to_string(w_out);
        kernelInfo.comp_options += " -DN=" + std::to_string(n);
        kernelInfo.comp_options += " -DC=" + std::to_string(c);
        kernelInfo.comp_options += " -DRD_BLCK=" + std::to_string(RD_BLCK);
        kernelInfo.comp_options += " -DHW_RD=" + std::to_string(HW_RD);
        kernelInfo.comp_options += " -DMAP_RD=" + std::to_string(MAP_RD);
        kernelInfo.comp_options += " -DREAD_TYPE=" + READ_TYPE;
        if (isBias) {
            kernelInfo.comp_options += " -DBIAS";
        }

        kernelInfo.l_wk = {lcl_size0, 1, 1};
        kernelInfo.g_wk = {MAP_RD, 1, 1};

        if (MAP_RD < MAX_ACTIVE_THREADS) {
            kernelInfo.g_wk = {MAP_RD, static_cast<size_t>(n), 1};
            kernelInfo.comp_options += " -D_2D_WG";
        }

        kptr = CreateKernel(device_id, &kernelInfo);
    } else {
        kernelInfo.kernel_name = "transpose_CNHW2NCHW";
        kernelInfo.kernel_type = MIOPEN;

        kernelInfo.comp_options = "";
        kernelInfo.comp_options += " -DNC_TRANS_CNHW";
        kernelInfo.comp_options += " -DN=" + std::to_string(n);
        kernelInfo.comp_options += " -DC=" + std::to_string(c);
        kernelInfo.comp_options += " -DHW_IN=" + std::to_string(h_in * w_in);
        kernelInfo.comp_options += " -DHW_OUT=" + std::to_string(h_out * w_out);
        kernelInfo.comp_options += " -DW_IN=" + std::to_string(w_in);
        kernelInfo.comp_options += " -DW_OUT=" + std::to_string(w_out);
        kernelInfo.comp_options += " -DH_STRIDE=" + std::to_string(h_stride);
        kernelInfo.comp_options += " -DW_STRIDE=" + std::to_string(w_stride);
        kernelInfo.comp_options += " -DIN_OFF=" + std::to_string(in_offset);
        kernelInfo.comp_options += " -DOUT_OFF=" + std::to_string(out_offset);

        size_t ld0 = WG_SIZE;
        size_t gd0 = c * h_out * w_out;

        kernelInfo.l_wk = {ld0, 1, 1};
        kernelInfo.g_wk = {gd0, 1, 1};

        if (gd0 < MAX_ACTIVE_THREADS) {
            kernelInfo.g_wk = {gd0, static_cast<size_t>(n), 1};
            kernelInfo.comp_options += " -D_2D_WG";
        }

        kptr = CreateKernel(device_id, &kernelInfo);
    }
}

} // namespace saber
} // namespace anakin
