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

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define SIZEOF_FLOAT 2 /* sizeof is unavailable for preprocessor */
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define SIZEOF_FLOAT 4
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#define UNUSED __attribute__((__unused__))

#ifndef MLO_FILTER_STRIDE0
#define MLO_FILTER_STRIDE0 1
#endif
#ifndef MLO_FILTER_STRIDE1
#define MLO_FILTER_STRIDE1 1
#endif

/// \todo Pass available LDS size to kernel during compilation.
#define MLO_LDS_MAX_SIZE 65536

#define MLO_FILTER_SZ (MLO_FILTER_SIZE1 * MLO_FILTER_SIZE0)

#define MLO_GRP_SZ0 (MLO_GRP_TILE0 * MLO_GRP_TILE1)
#define MLO_GRP_SZ1 1
#define MLO_GRP_SZ2 1
#define MLO_GRP_SZ (MLO_GRP_SZ0 * MLO_GRP_SZ1 * MLO_GRP_SZ2)
#define MLO_N_PROC_WAVES ((MLO_GRP_SZ + MLO_N_READ_PROCS - 1) / MLO_N_READ_PROCS)
#define MLO_OUT_TILE_SZ (MLO_OUT_TILE1 * MLO_OUT_TILE0)
#define MLO_ALU_TILE_SZ (MLO_ALU_VTILE1 * MLO_ALU_VTILE0)

#if MLO_IN_TILE0 < MLO_OUT_WIDTH || MLO_IN_TILE1 < MLO_OUT_HEIGHT
#define MLO_LARGE_MAP 1
#else
#define MLO_LARGE_MAP 0
#endif

#if(MLO_IN_WIDTH == MLO_OUT_WIDTH &&                                \
    (MLO_IN_WIDTH / MLO_IN_TILE0) * MLO_IN_TILE0 == MLO_IN_WIDTH && \
    MLO_IN_HEIGHT == MLO_OUT_HEIGHT &&                              \
    (MLO_IN_HEIGHT / MLO_IN_TILE1) * MLO_IN_TILE1 == MLO_IN_HEIGHT)
#define MLO_OUT_ALIGNED 1
#else
#define MLO_OUT_ALIGNED 0
#endif

#define MLO_N_ALUTILES_TOTAL ((MLO_GRP_TILE0 * MLO_GRP_TILE1) / (MLO_ALU_TILE_SZ))
#define MLO_N_ALUTILES_PERSTACK (MLO_N_ALUTILES_TOTAL / MLO_N_STACKS)
#define MLO_ALUTILES_STACK_SZ (MLO_N_ALUTILES_PERSTACK * MLO_ALU_TILE_SZ)
#define MLO_N_IN_TILES_TOTAL (MLO_N_IN_TILES_PERSTACK * MLO_N_STACKS)
/*
#define MLO_N_OUT_TILES_PERSTACK (MLO_N_OUT_TILES*MLO_N_ALUTILES_PERSTACK)
#if MLO_N_OUT_TILES_PERSTACK > MLO_N_OUTPUTS
#undef MLO_N_OUT_TILES_PERSTACK
#define MLO_N_OUT_TILES_PERSTACK MLO_N_OUTPUTS
#endif
*/
#define MLO_N_OUT_TILE_BLOCKS0 ((MLO_OUT_WIDTH + MLO_IN_TILE0 - 1) / MLO_IN_TILE0)
#define MLO_N_OUT_TILE_BLOCKS1 ((MLO_OUT_HEIGHT + MLO_IN_TILE1 - 1) / MLO_IN_TILE1)
#define MLO_N_IN_PACKS ((MLO_N_INPUTS + MLO_N_IN_TILES_PERSTACK - 1) / MLO_N_IN_TILES_PERSTACK)

#define MLO_N_IN_READ (MLO_N_IN_PACKS * MLO_N_IN_TILES_PERSTACK)
#if MLO_N_IN_READ == MLO_N_INPUTS
#define MLO_INPUTS_ALIGNED 1
#else
#define MLO_INPUTS_ALIGNED 0
#endif

#define MLO_N_OUT_PACKS (MLO_N_OUTPUTS / MLO_N_OUT_TILES_PERSTACK)
#if MLO_N_OUT_PACKS * MLO_N_OUT_TILES_PERSTACK == MLO_N_OUTPUTS && \
    MLO_N_OUT_TILES_PERSTACK != MLO_N_OUTPUTS
#define MLO_OUTPUTS_ALIGNED 1
#else
#define MLO_OUTPUTS_ALIGNED 0
#endif

#define MLO_N_BATCH_PACKS (MLO_BATCH_SZ / MLO_N_STACKS)
#if MLO_N_BATCH_PACKS * MLO_N_STACKS == MLO_BATCH_SZ && MLO_N_STACKS != MLO_BATCH_SZ
#define MLO_BATCH_ALIGNED 1
#else
#define MLO_BATCH_ALIGNED 0
#endif

#if MLO_DIR_FORWARD == 1
// here we use kernel size. it's
// important when padding == 0  2*
// MLO_FILTER_PAD0
#define MLO_IN_LCL_WIDTH ((MLO_IN_TILE0 - 1) * MLO_FILTER_STRIDE0 + (MLO_FILTER_SIZE0 - 1) * MLO_FILTER_DILATION0 + 1)
#define MLO_IN_LCL_HEIGHT ((MLO_IN_TILE1 - 1) * MLO_FILTER_STRIDE1 + (MLO_FILTER_SIZE1 - 1) * MLO_FILTER_DILATION1 + 1)
#else
#define MLO_IN_LCL_WIDTH                                              \
    ((MLO_IN_TILE0 + MLO_FILTER_SIZE0 - 1 + MLO_FILTER_STRIDE0 - 1) / \
     MLO_FILTER_STRIDE0) // here we use kernel size. it's important when padding == 0  2*
// MLO_FILTER_PAD0
#define MLO_IN_LCL_HEIGHT \
    ((MLO_IN_TILE1 + MLO_FILTER_SIZE1 - 1 + MLO_FILTER_STRIDE1 - 1) / MLO_FILTER_STRIDE1)
#endif
#define MLO_IN_LCL_TILE_SZ (MLO_IN_LCL_WIDTH * MLO_IN_LCL_HEIGHT)
#define MLO_IN_LCL_PERSTACK_SZ (MLO_IN_LCL_TILE_SZ * MLO_N_IN_TILES_PERSTACK)
#define MLO_IN_LCL_SZ (MLO_IN_LCL_PERSTACK_SZ * MLO_N_STACKS)

#define MLO_WEIGHTS_SZ (MLO_N_OUT_TILES_PERSTACK * MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ)

#define MLO_PVT_ACCUM_DATA_SZ (MLO_N_OUT_TILES * MLO_OUT_TILE_SZ)
#if MLO_DIR_FORWARD == 1
#define MLO_PVT_IN_WIDTH MLO_FILTER_SIZE0
#define MLO_PVT_IN_HEIGHT MLO_FILTER_SIZE1
#else
#define MLO_PVT_IN_WIDTH \
    ((MLO_OUT_TILE0 + MLO_FILTER_SIZE0 - 1 + MLO_FILTER_STRIDE0 - 1) / MLO_FILTER_STRIDE0)
#define MLO_PVT_IN_HEIGHT ((MLO_OUT_TILE1 + MLO_FILTER_STRIDE1 - 1) / MLO_FILTER_STRIDE1)
#endif

#define MLO_LCL_WEIGHTS 1

#define MLO_PADDING_SHIFT1 (MLO_FILTER_SIZE1 - MLO_FILTER_PAD1 - 1)
#define MLO_PADDING_SHIFT0 (MLO_FILTER_SIZE0 - MLO_FILTER_PAD0 - 1)

#define MLO_PADDING_FIX1 (MLO_FILTER_SIZE1 % MLO_OUT_TILE1)
#define MLO_PADDING_FIX0 (MLO_FILTER_SIZE0 % MLO_OUT_TILE0)

#if defined(__AMDGCN__)
extern uint __llvm_amdgcn_readfirstlane(uint) __asm("llvm.amdgcn.readfirstlane");
#define uniform(x) __llvm_amdgcn_readfirstlane(x)
#else
#define uniform(x) (x)
#endif

static inline uint iDiv(uint v, uint d) {
    uint r = v / d;
    return (r);
}

static inline uint iMod(uint v, uint u, uint d) {
    uint r = v - u * d;
    return (r);
}

static inline void calculateXYPos(uint linPos, uint width, uint* __restrict x, uint* __restrict y) {
    (*y) = linPos / width;
    (*x) = linPos - (*y) * width;
}

static inline uint calculateOffset(uint stride, uint x, uint y) {
    uint ret = y * stride + x;
    return (ret);
}

static inline void readDataElem(uint linPos,
                                __local _FLOAT* lcl_data,
                                uint lcl_base,
                                UNUSED uint lcl_height,
                                uint lcl_width,
                                uint lcl_stride,
                                uint lcl_y,
                                uint lcl_x,
                                const __global _FLOAT* gbl_data,
                                uint gbl_base,
                                uint gbl_height,
                                uint gbl_width,
                                uint gbl_stride,
                                uint gbl_y,
                                uint gbl_x,
                                bool vis,
                                UNUSED bool debug) {
    uint x, y;
    calculateXYPos(linPos, lcl_width, &x, &y);
    uint g_x      = x + gbl_x;
    uint g_y      = y + gbl_y;
    uint gbl_off0 = calculateOffset(gbl_stride, g_x, g_y);
    uint gbl_off  = gbl_off0 + gbl_base;

#if MLO_LARGE_MAP == 1
    uint lcl_off = lcl_base + linPos;
    (void)lcl_stride;
    (void)lcl_x;
    (void)lcl_y;
#else
    uint l_x     = x + lcl_x;
    uint l_y     = y + lcl_y;
    uint lcl_off = lcl_base + l_y * lcl_stride + l_x;
#endif

#if MLO_LARGE_MAP == 1
    //  vis &= (g_x >= 0 && g_x < gbl_width && g_y >= 0 && g_y < gbl_height);
    vis &= (g_x < gbl_width && g_y < gbl_height);
#else
    (void)gbl_width;
    (void)gbl_height;
#endif
    gbl_off        = (vis) ? gbl_off : 0;
    _FLOAT gbl_val = gbl_data[gbl_off];
    gbl_val        = (vis) ? gbl_val : 0;

    lcl_data[lcl_off] = gbl_val;
}

static inline void readData(uint lcl_id,
                            uint size,
                            uint lcl_p_stride,
                            __local _FLOAT* lcl_data,
                            uint lcl_base,
                            uint lcl_height,
                            uint lcl_width,
                            uint lcl_stride,
                            uint lcl_y,
                            uint lcl_x,
                            const __global _FLOAT* gbl_data,
                            uint gbl_base,
                            uint gbl_height,
                            uint gbl_width,
                            uint gbl_stride,
                            uint gbl_y,
                            uint gbl_x,
                            bool vis,
                            bool debug) {

    for (uint i = lcl_id; i < size; i += lcl_p_stride) {
        readDataElem(i,
                     lcl_data,
                     lcl_base,
                     lcl_height,
                     lcl_width,
                     lcl_stride,
                     lcl_y,
                     lcl_x,
                     gbl_data,
                     gbl_base,
                     gbl_height,
                     gbl_width,
                     gbl_stride,
                     gbl_y,
                     gbl_x,
                     vis,
                     debug);
    }
}

static inline void loadData(uint lcl_id,
                            uint lcl_p_stride,
                            __local _FLOAT* lcl_data,
                            uint lcl_off,
                            uint lcl_size,
                            uint lcl_height,
                            uint lcl_width,
                            uint lcl_stride,
                            uint lcl_bot_y,
                            uint lcl_bot_x,
                            const __global _FLOAT* gbl_data,
                            uint gbl_off,
                            uint gbl_size,
                            uint gbl_height,
                            uint glb_width,
                            uint gbl_stride,
                            uint gbl_bot_y,
                            uint gbl_bot_x,
                            uint buf_block_ind,
                            uint max_n_bufs,
                            uint lcl_n_bufs,
                            bool debug) {

    for (uint c = 0; c < lcl_n_bufs; ++c, lcl_off += lcl_size, gbl_off += gbl_size) {
        bool vis = (buf_block_ind + c < max_n_bufs);
        readData(lcl_id,
                 lcl_size,
                 lcl_p_stride,
                 lcl_data,
                 lcl_off,
                 lcl_height,
                 lcl_width,
                 lcl_stride,
                 lcl_bot_y,
                 lcl_bot_x,
                 gbl_data,
                 gbl_off,
                 gbl_height,
                 glb_width,
                 gbl_stride,
                 gbl_bot_y,
                 gbl_bot_x,
                 vis,
                 (debug));
    }
}

static inline void Conv(uint o_map_base,
                        uint in_stg_off,
                        __private _FLOAT* __restrict pvt_in_stage,
                        __local _FLOAT* __restrict lcl_indata,
                        __local _FLOAT* __restrict lcl_wei,
                        __private _FLOAT* __restrict pvt_accum) {
    // preload input
    uint in_stg_off1 = in_stg_off;
    uint in_pvt_off = 0;
#pragma unroll
    for (uint i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK; ++i_c, in_stg_off1 += MLO_IN_LCL_TILE_SZ, in_pvt_off += MLO_FILTER_SZ) {
        int in_stg_off2 = in_stg_off1;
        uint in_pvt_off2 = in_pvt_off;
#pragma unroll
        for (int j = 0; j < MLO_PVT_IN_HEIGHT; ++j, in_stg_off2 += MLO_IN_LCL_WIDTH * MLO_FILTER_DILATION1, in_pvt_off2 += MLO_FILTER_SIZE0) {
#pragma unroll
            for (uint i = 0; i < MLO_PVT_IN_WIDTH; ++i) {
                pvt_in_stage[in_pvt_off2 + i] = lcl_indata[in_stg_off2 + i*MLO_FILTER_DILATION0];
            }
        }
    }

    // over filter rows
    uint wei_stg_base_off = o_map_base * MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ;
#pragma unroll
    for (uint o_c = 0; o_c < MLO_N_OUT_TILES; ++o_c, wei_stg_base_off += MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) {
        uint wei_stg_off = wei_stg_base_off;
        uint in_pvt_off = 0;
#pragma unroll
        for (uint i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK; ++i_c, wei_stg_off += MLO_FILTER_SZ, in_pvt_off += MLO_FILTER_SZ) {
            uint wei_stg_off2 = wei_stg_off;
            uint in_pvt_off2 = in_pvt_off;
#pragma unroll
            for (uint k = 0; k < MLO_FILTER_SIZE1; ++k, wei_stg_off2 += MLO_FILTER_SIZE0, in_pvt_off2 += MLO_FILTER_SIZE0) {
#pragma unroll
                for (uint l = 0; l < MLO_FILTER_SIZE0; ++l) {
                    pvt_accum[o_c] += pvt_in_stage[in_pvt_off2 + l] * lcl_wei[wei_stg_off2 + l];
                }
            } // for(uint o_c = 0; o_c < MLO_N_OUT_TILES; ++o_c)
        } // for(uint k = 0; k < MLO_FILER_SIZE1; ++k,in_stg_off2+=MLO_IN_LCL_WIDTH)
    } // for(uint i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK; ++i_c, in_stg_off1 +=
}

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenGroupConvUni(const __global _FLOAT* __restrict in,
                   const __global _FLOAT* __restrict weights,
#if MLO_CONV_BIAS
                   const __global _FLOAT* __restrict bias,
#endif
                   __global _FLOAT* __restrict out,
#if MLO_WITH_RELU
                   _FLOAT slope,
#endif
                   UNUSED _FLOAT padding_val) {
#if((MLO_IN_LCL_SZ + MLO_WEIGHTS_SZ) * SIZEOF_FLOAT) > MLO_LDS_MAX_SIZE
#error "Local memory size should not exceed 64k."
#endif
    __local _FLOAT lcl_indata[MLO_IN_LCL_SZ];
    __local _FLOAT lcl_wei[MLO_WEIGHTS_SZ];
    __private _FLOAT pvt_accum[MLO_PVT_ACCUM_DATA_SZ];
    __private _FLOAT pvt_in_stage[MLO_N_IN_TILES_PERSTACK * MLO_PVT_IN_HEIGHT * MLO_PVT_IN_WIDTH];


    uint grp_id0 = get_group_id(0);
#if MLO_N_OUT_TILE_BLOCKS0 & (MLO_N_OUT_TILE_BLOCKS0 - 1)
    uint y_tile_blk = iDiv(grp_id0, MLO_N_OUT_TILE_BLOCKS0);
    uint x_tile_blk = iMod(grp_id0, y_tile_blk, MLO_N_OUT_TILE_BLOCKS0);
#else
    uint y_tile_blk       = grp_id0 / MLO_N_OUT_TILE_BLOCKS0;
    uint x_tile_blk       = grp_id0 & (MLO_N_OUT_TILE_BLOCKS0 - 1);
#endif
    uint o_pack = get_group_id(1); // block of outputs
    uint b_pack = get_group_id(2); // batch block

    uint lcl_id = get_local_id(0);

#if MLO_ALUTILES_STACK_SZ >= MLO_GRP_SZ
    uint stack        = 0;
    uint alu_stack_id = lcl_id;
#elif MLO_ALUTILES_STACK_SZ & (MLO_ALUTILES_STACK_SZ - 1)
    uint stack            = iDiv(lcl_id, MLO_ALUTILES_STACK_SZ);        // stack
    uint alu_stack_id     = iMod(lcl_id, stack, MLO_ALUTILES_STACK_SZ); // alu index in stack
#else
    uint stack = lcl_id / MLO_ALUTILES_STACK_SZ; // stack
    uint alu_stack_id = lcl_id & (MLO_ALUTILES_STACK_SZ - 1); // alu index in stack
#if MLO_ALUTILES_STACK_SZ >= 64
    stack = uniform(stack);
#endif
#endif
    // ALU plane inside stack
#if MLO_ALU_TILE_SZ & (MLO_ALU_TILE_SZ - 1)
    uint alu_out_plane_id = iDiv(alu_stack_id, MLO_ALU_TILE_SZ); // alu output plane index
    uint alu_out_id       = iMod(
                                alu_stack_id, alu_out_plane_id, MLO_ALU_TILE_SZ); // alu index inside an ALU output plane
#else
    uint alu_out_plane_id = alu_stack_id / MLO_ALU_TILE_SZ;             // alu output plane index
    uint alu_out_id       = alu_stack_id & (MLO_ALU_TILE_SZ -
                                            1);       // alu index inside an ALU output plane
#endif
    // pos inside ALU tile
#if MLO_ALU_VTILE0 & (MLO_ALU_VTILE0 - 1)
    uint alu_tl1 = iDiv(alu_out_id, MLO_ALU_VTILE0);
    uint alu_tl0 = iMod(alu_out_id, alu_tl1, MLO_ALU_VTILE0);
#else
    uint alu_tl1          = alu_out_id / MLO_ALU_VTILE0;
    uint alu_tl0          = alu_out_id & (MLO_ALU_VTILE0 - 1);
#endif

    uint ig          = o_pack / MLO_STACK_PERGROUP;
    uint within_ig   = o_pack % MLO_STACK_PERGROUP;
    uint o_map_plane = ig * MLO_GROUP_TILES + within_ig * MLO_N_OUT_TILES_PERSTACK;
    uint o_map_base  = alu_out_plane_id * MLO_N_OUT_TILES; // local output map offset
    uint o_map       = o_map_plane + o_map_base;           // output map index per ALU plane
    uint b_index     = b_pack * MLO_N_STACKS;

#if MLO_LARGE_MAP != 1
#if MLO_N_READ_PROCS >= MLO_GRP_SZ
    uint wave_id     = 0;
    uint wave_lcl_id = lcl_id;
#elif MLO_N_READ_PROCS & (MLO_N_READ_PROCS - 1)
    uint wave_id     = iDiv(lcl_id, MLO_N_READ_PROCS);
    uint wave_lcl_id = iMod(lcl_id, wave_id, MLO_N_READ_PROCS);
#else
    uint wave_id     = lcl_id / MLO_N_READ_PROCS;
    uint wave_lcl_id = lcl_id & (MLO_N_READ_PROCS - 1);
#if MLO_N_READ_PROCS >= 64
    wave_id          = uniform(wave_id);
#endif
#endif
#endif

#if MLO_DIR_FORWARD == 1
    uint x_grp = x_tile_blk * MLO_IN_TILE0 * MLO_FILTER_STRIDE0;
    uint y_grp = y_tile_blk * MLO_IN_TILE1 * MLO_FILTER_STRIDE1;
#if MLO_LARGE_MAP == 1
    uint x_in_grp = x_grp - MLO_FILTER_PAD0;
    uint y_in_grp = y_grp - MLO_FILTER_PAD1;
#endif
    uint x_in_lcl = alu_tl0 * MLO_OUT_TILE0 * MLO_FILTER_STRIDE0;
    uint y_in_lcl = alu_tl1 * MLO_OUT_TILE1 * MLO_FILTER_STRIDE1;
#else
    uint x_grp            = x_tile_blk * (MLO_IN_TILE0 / MLO_FILTER_STRIDE0);
    uint y_grp            = y_tile_blk * (MLO_IN_TILE1 / MLO_FILTER_STRIDE1);
#if MLO_LARGE_MAP == 1
    uint x_in_grp         = x_grp - (MLO_FILTER_PAD0 / MLO_FILTER_STRIDE0);
    uint y_in_grp         = y_grp - (MLO_FILTER_PAD1 / MLO_FILTER_STRIDE1);
#endif
    uint x_in_lcl         = alu_tl0 * (MLO_OUT_TILE0 / MLO_FILTER_STRIDE0);
    uint y_in_lcl         = alu_tl1 * (MLO_OUT_TILE1 / MLO_FILTER_STRIDE1);
#endif

    // base offset to read data from local input data
    uint in_stg_off = stack * MLO_IN_LCL_PERSTACK_SZ + (y_in_lcl) * MLO_IN_LCL_WIDTH + x_in_lcl;

#if MLO_LARGE_MAP == 0

    for (uint i = lcl_id; i < MLO_IN_LCL_SZ; i += MLO_GRP_SZ) {
        lcl_indata[i] = 0;
    }

#endif

    for (uint i = 0; i < MLO_PVT_ACCUM_DATA_SZ; ++i) {
        pvt_accum[i] = 0;
    }

#if MLO_DIR_FORWARD == 1
    uint wei_off0 = ((MLO_N_OUTPUTS / MLO_GROUP_COUNTS) * ig +
                     ((o_map % (MLO_N_OUTPUTS / MLO_GROUP_COUNTS)) / MLO_N_OUT_TILES_PERSTACK) *
                     MLO_N_OUT_TILES_PERSTACK) *
                    (MLO_N_INPUTS / MLO_GROUP_COUNTS) * MLO_FILTER_SZ;
#else
    uint wei_off0 = ((MLO_N_OUTPUTS / MLO_GROUP_COUNTS) * (MLO_N_INPUTS / MLO_GROUP_COUNTS) * ig +
                     ((o_map % (MLO_N_OUTPUTS / MLO_GROUP_COUNTS)) / MLO_N_OUT_TILES_PERSTACK) *
                     MLO_N_OUT_TILES_PERSTACK) *
                    MLO_FILTER_SZ;
#endif

    uint in_off0 = (MLO_N_INPUTS / MLO_GROUP_COUNTS) * ig * MLO_IN_CHANNEL_STRIDE +
                   b_index * MLO_IN_BATCH_STRIDE;

    for (uint ic = (MLO_N_INPUTS / MLO_GROUP_COUNTS) * ig;
            ic < (MLO_N_INPUTS / MLO_GROUP_COUNTS) * (ig + 1);
            ic += MLO_N_IN_TILES_PERSTACK,
            in_off0 += MLO_IN_CHANNEL_STRIDE * MLO_N_IN_TILES_PERSTACK,
            wei_off0 += MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ
#if MLO_DIR_FORWARD == 0
                        *
                        (MLO_N_OUTPUTS / MLO_GROUP_COUNTS)
#endif
        ) {
        barrier(CLK_LOCAL_MEM_FENCE);

        // small map has been read in full continiously into the lDS buffer within padded rect,
        // padding has been done on initilization.
        // large map calculates padding on the fly and fills it with 0.

#if 1 // all inputs

#if MLO_LARGE_MAP == 1
        uint in_lcl_off1 = 0;
        uint in_off1     = in_off0;

        for (uint i_b = 0; i_b < MLO_N_STACKS;
                ++i_b, in_off1 += MLO_IN_BATCH_STRIDE, in_lcl_off1 += MLO_IN_LCL_PERSTACK_SZ) {
            bool vis = true;
#if MLO_BATCH_ALIGNED == 0
            vis &= (b_index + i_b < MLO_BATCH_SZ);
#endif

            // over all inputs in stack
            uint in_off2     = in_off1;
            uint in_lcl_off2 = in_lcl_off1;

            for (uint i_c = 0; i_c < MLO_N_IN_TILES_PERSTACK;
                    ++i_c, in_off2 += MLO_IN_CHANNEL_STRIDE, in_lcl_off2 += MLO_IN_LCL_TILE_SZ) {
                vis &= (ig < MLO_GROUP_COUNTS);
                vis &= (ic + i_c < (MLO_N_INPUTS / MLO_GROUP_COUNTS) * (ig + 1));
                vis &= (ic + i_c >= (MLO_N_INPUTS / MLO_GROUP_COUNTS) * ig);

                uint elem_id      = lcl_id;
                uint lcl_p_stride = MLO_GRP_SZ0;
                uint lcl_base     = 0;
                uint lcl_y        = 0;
                uint lcl_x        = 0;
                uint gbl_base     = in_off2;

                readData(elem_id,
                         (MLO_IN_LCL_HEIGHT * MLO_IN_LCL_WIDTH),
                         lcl_p_stride,
                         &lcl_indata[in_lcl_off2],
                         lcl_base,
                         MLO_IN_LCL_HEIGHT,
                         MLO_IN_LCL_WIDTH,
                         MLO_IN_LCL_WIDTH,
                         lcl_y,
                         lcl_x,
                         &in[0],
                         gbl_base,
                         MLO_IN_HEIGHT,
                         MLO_IN_WIDTH,
                         MLO_IN_STRIDE,
                         y_in_grp,
                         x_in_grp,
                         vis,
                         true);
            }
        }

#else

        for (uint i = wave_id; i < MLO_N_IN_TILES_TOTAL; i += MLO_N_PROC_WAVES) {
#if MLO_N_IN_TILES_PERSTACK & (MLO_N_IN_TILES_PERSTACK - 1)
            uint i_b = iDiv(i, MLO_N_IN_TILES_PERSTACK);
            uint i_c = iMod(i, i_b, MLO_N_IN_TILES_PERSTACK);
#else
            uint i_b   = i / MLO_N_IN_TILES_PERSTACK;
            uint i_c   = i & (MLO_N_IN_TILES_PERSTACK - 1);
#endif

            bool vis = true;

#if MLO_BATCH_ALIGNED == 0
            vis &= (b_index + i_b < MLO_BATCH_SZ);
#endif

            vis &= (ig < MLO_GROUP_COUNTS);
            vis &= (ic + i_c < (MLO_N_INPUTS / MLO_GROUP_COUNTS) * (ig + 1));
            vis &= (ic + i_c >= (MLO_N_INPUTS / MLO_GROUP_COUNTS) * ig);

            uint in_off2     = in_off0 + i_b * MLO_IN_BATCH_STRIDE + i_c * MLO_IN_CHANNEL_STRIDE;
            uint in_lcl_off2 = i_b * MLO_IN_LCL_PERSTACK_SZ + i_c * MLO_IN_LCL_TILE_SZ;

            uint elem_id      = wave_lcl_id;
            uint lcl_p_stride = MLO_N_READ_PROCS;
            uint lcl_base     = 0;
#if MLO_DIR_FORWARD == 1
            uint lcl_y        = MLO_FILTER_PAD1;
            uint lcl_x        = MLO_FILTER_PAD0;
#else
            uint lcl_y = (MLO_FILTER_PAD1 / MLO_FILTER_STRIDE0);
            uint lcl_x = (MLO_FILTER_PAD0 / MLO_FILTER_STRIDE1);
#endif
            uint gbl_base     = in_off2;

            readData(elem_id,
                     (MLO_IN_HEIGHT * MLO_IN_WIDTH),
                     lcl_p_stride,
                     &lcl_indata[in_lcl_off2],
                     lcl_base,
                     MLO_IN_HEIGHT,
                     MLO_IN_WIDTH,
                     MLO_IN_LCL_WIDTH,
                     lcl_y,
                     lcl_x,
                     &in[0],
                     gbl_base,
                     MLO_IN_HEIGHT,
                     MLO_IN_WIDTH,
                     MLO_IN_STRIDE,
                     y_grp,
                     x_grp,
                     vis,
                     true);
        }

#endif

        // read inputs and weights
        // put weights into LDS

#if 1 // only weights

        for (uint i = lcl_id; i < MLO_WEIGHTS_SZ; i += MLO_GRP_SZ) {
#if MLO_DIR_FORWARD == 1
            // here is [tops][bottoms]
#if(MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) & ((MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) - 1)
            uint lcl_o = iDiv(i, (MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ));
            uint gbl_i = iMod(i, lcl_o, (MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ));
#else
            uint lcl_o = i / (MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ);
            uint gbl_i = i & ((MLO_N_IN_TILES_PERSTACK * MLO_FILTER_SZ) - 1);
#endif
            uint gbl_we_off =
                wei_off0 + lcl_o * (MLO_N_INPUTS / MLO_GROUP_COUNTS) * MLO_FILTER_SZ + gbl_i;

            bool within_range =
                gbl_we_off < ((MLO_N_OUTPUTS / MLO_GROUP_COUNTS) *
                              (MLO_N_INPUTS / MLO_GROUP_COUNTS) * (ig + 1) * MLO_FILTER_SZ);
            within_range &= gbl_we_off >= ((MLO_N_OUTPUTS / MLO_GROUP_COUNTS) *
                                           (MLO_N_INPUTS / MLO_GROUP_COUNTS) * ig * MLO_FILTER_SZ);
            within_range &= (ig < MLO_GROUP_COUNTS);

            gbl_we_off = (within_range) ? gbl_we_off : 0;
            _FLOAT wei = weights[gbl_we_off];
            wei        = (within_range) ? wei : 0;
            lcl_wei[i] = wei;
#else
            // outputs are botoms(inputs))
            // inputs are tops(outputs)
#if(MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) & ((MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) - 1)
            uint lcl_o = iDiv(i, (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ));
            uint gbl_i = iMod(i, lcl_o, (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ));
#else
            uint lcl_o = i / (MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ);
            uint gbl_i = i & ((MLO_N_OUT_TILES_PERSTACK * MLO_FILTER_SZ) - 1);
#endif
#if MLO_FILTER_SZ & (MLO_FILTER_SZ - 1)
            uint lcl_c = iDiv(gbl_i, MLO_FILTER_SZ);
            uint lcl_i = iMod(gbl_i, lcl_c, MLO_FILTER_SZ);
#else
            uint lcl_c = gbl_i / MLO_FILTER_SZ;
            uint lcl_i = gbl_i & (MLO_FILTER_SZ - 1);
#endif

            uint lcl_we_off = (lcl_c * MLO_N_IN_TILES_PERSTACK + lcl_o) * MLO_FILTER_SZ + lcl_i;
            uint gbl_we_off = (lcl_o * (MLO_N_OUTPUTS / MLO_GROUP_COUNTS) + lcl_c) * MLO_FILTER_SZ + wei_off0 + lcl_i;
            bool within_range =
                gbl_we_off < ((MLO_N_OUTPUTS / MLO_GROUP_COUNTS) *
                              (MLO_N_INPUTS / MLO_GROUP_COUNTS) * (ig + 1) * MLO_FILTER_SZ);
            within_range &= gbl_we_off >= ((MLO_N_OUTPUTS / MLO_GROUP_COUNTS) *
                                           (MLO_N_INPUTS / MLO_GROUP_COUNTS) * ig * MLO_FILTER_SZ);
            within_range &= (ig < MLO_GROUP_COUNTS);

            gbl_we_off          = (within_range) ? gbl_we_off : 0;
            _FLOAT wei          = weights[gbl_we_off];
            wei                 = (within_range) ? wei : 0;
            lcl_wei[lcl_we_off] = wei;
#endif
        }

#endif

        // over all batch stacks

#endif // all input

        barrier(CLK_LOCAL_MEM_FENCE);

        // convolution
        Conv(o_map_base, in_stg_off, pvt_in_stage, lcl_indata, lcl_wei, pvt_accum);

        //      barrier(CLK_LOCAL_MEM_FENCE);
    }

#if 1

    // write results out
#if MLO_DIR_FORWARD == 1
#if MLO_FILTER_STRIDE0 == 1
    uint x_out_grp = x_grp;
#else
    uint x_out_grp = x_tile_blk * MLO_IN_TILE0;
#endif
#if MLO_FILTER_STRIDE1 == 1
    uint y_out_grp = y_grp;
#else
    uint y_out_grp = y_tile_blk * MLO_IN_TILE1;
#endif
#else
    uint x_out_grp = x_grp * MLO_FILTER_STRIDE0;
    uint y_out_grp = y_grp * MLO_FILTER_STRIDE1;
#endif
    uint x_out_lcl = alu_tl0 * MLO_OUT_TILE0;
    uint y_out_lcl = alu_tl1 * MLO_OUT_TILE1;

    uint out_off = (b_index + stack) * MLO_OUT_BATCH_STRIDE + o_map * MLO_OUT_CHANNEL_STRIDE +
                   (y_out_grp + y_out_lcl) * MLO_OUT_STRIDE + x_out_grp + x_out_lcl;
    // over all local stacks
#if MLO_BATCH_ALIGNED == 0

    if (b_index + stack < MLO_BATCH_SZ)
#endif
    {

        // over all local outputs
        uint out_off1 = out_off;

        for (uint o = 0; o < MLO_N_OUT_TILES; ++o, out_off1 += MLO_OUT_CHANNEL_STRIDE) {
            if (o_map + o < (MLO_N_OUTPUTS / MLO_GROUP_COUNTS) * (ig + 1) &&
                    o_map + o >= (MLO_N_OUTPUTS / MLO_GROUP_COUNTS) * ig && ig < MLO_GROUP_COUNTS) {
                // over output tile
                uint out_off2 = out_off1;
#if MLO_OUT_TILE0 == 1

                for (uint j = 0; j < MLO_OUT_TILE1 && y_out_grp + y_out_lcl + j < MLO_OUT_HEIGHT;
                        ++j, out_off2 += MLO_OUT_STRIDE) {
                    for (uint i = 0;
                            i < MLO_OUT_TILE0 && x_out_grp + x_out_lcl + i < MLO_OUT_WIDTH &&
                            out_off2 + i < MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ;
                            ++i) {
#else

                for (uint j = 0; j < MLO_OUT_TILE1; ++j, out_off2 += MLO_OUT_STRIDE) {

                    if (y_out_grp + y_out_lcl + j < MLO_OUT_HEIGHT) {
                        for (uint i = 0; i < MLO_OUT_TILE0; ++i) {
                            if (x_out_grp + x_out_lcl + i < MLO_OUT_WIDTH &&
                                    out_off2 + i < MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ) {
#endif
                        out[out_off2 + i] = pvt_accum[o * MLO_OUT_TILE_SZ + j * MLO_OUT_TILE0 + i]
#if MLO_CONV_BIAS
                                            + bias[o_map + o]
#endif
                                            ;

#if MLO_WITH_RELU
                        //ReLU fusion
                        out[out_off2 + i] *= out[out_off2 + i] > 0.0f ? 1.0f : slope;
#endif

#if MLO_OUT_TILE0 != 1
                            }
                        }

#endif
                    }
                }
            }
        }
    }
#endif
}
