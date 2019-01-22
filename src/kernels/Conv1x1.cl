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

#ifdef MACRO

#ifndef KERNEL_METHOD
#define KERNEL_METHOD   1
#endif


#if KERNEL_METHOD == 1

#if GLOBAL_SPLITU == 1
#define ATOMIC  0
#else
#define ATOMIC  2
#endif

///////////////////////////////////////////
#ifndef LOCAL_X
#define LOCAL_X     256
#endif

#ifndef METHOD
#define METHOD  1
#endif

#ifndef BRANCH
#if N == 1
#define BRANCH  1
#elif N == 2
#define BRANCH  3
#endif
#endif

#define GSU_MINUS_ONE   (GLOBAL_SPLITU - 1)
#define BARRIER_MARK    (0xffffffff)
#define STRIDE_WEI  (C)
#define ALIGNED_C   (C + GLOBAL_SPLITU * PER_ITER_STRIDE - 1) / (GLOBAL_SPLITU * PER_ITER_STRIDE) * (GLOBAL_SPLITU * PER_ITER_STRIDE)
#if STRIDE == 2
#define OW  (W / 2)
#define OH  (H / 2)
#define STRIDE_IN_REAL  (H * W)
#else
#define OW  (W)
#define OH  (H)
#endif
#define STRIDE_IN   (OH * OW)


#define WG_PER_IN   ((STRIDE_IN + TILE_COL - 1) / TILE_COL)
#define WG_PER_WEI  ((K + TILE_ROW - 1) / TILE_ROW)

#define QSTRIDE (ALIGNED_C / GLOBAL_SPLITU)
#define ITER    (QSTRIDE / PER_ITER_STRIDE)

#if BRANCH == 1 || BRANCH == 2
#define LDS_WEI_COL (PER_ITER_STRIDE)
#define LDS_WEI_STRIDE  (LDS_WEI_COL + (LDS_WEI_COL + 31) / 32)
#define LDS_WEI_ROW (TILE_ROW)
#define WEI_READ_LINE   LDS_WEI_COL
#elif BRANCH == 3 || BRANCH == 4
#define LDS_WEI_COL (TILE_ROW)
#define LDS_WEI_STRIDE  (LDS_WEI_COL + (LDS_WEI_COL + 31) / 32)
#define LDS_WEI_ROW (PER_ITER_STRIDE)
#define WEI_READ_LINE   LDS_WEI_ROW
#endif
#if BRANCH == 1 || BRANCH == 3
#define LDS_IN_COL  (PER_ITER_STRIDE)
#define LDS_IN_STRIDE   (LDS_IN_COL + (LDS_IN_COL + 31) / 32)
#define LDS_IN_ROW  (TILE_COL)
#define IN_READ_LINE    LDS_IN_ROW
#elif BRANCH == 2 || BRANCH == 4
#define LDS_IN_COL  (TILE_COL)
#define LDS_IN_STRIDE   (LDS_IN_COL + (LDS_IN_COL + 31) / 32)
#define LDS_IN_ROW  (PER_ITER_STRIDE)
#define IN_READ_LINE    LDS_IN_COL
#endif

#define LDS_WEI_READ_ITER   (LDS_WEI_COL * LDS_WEI_ROW / LOCAL_X)
#define LDS_IN_READ_ITER    (LDS_IN_COL * LDS_IN_ROW / LOCAL_X)
#define LDS_WEI_ROW_START   ((lid_x / WEI_READ_LINE) * LDS_WEI_READ_ITER)
#define LDS_IN_ROW_START    ((lid_x / IN_READ_LINE) * LDS_IN_READ_ITER)

#define IN_BATCH_STRIDE     (C * H * W)
#define OUT_BATCH_STRIDE    (K * OH * OW)

#define COUNTER_STRIDE  (WG_PER_IN * WG_PER_WEI)
#define GROUP_ITER  (grid_x / COUNTER_STRIDE)
#define WEI_ROW_START   ((grid_x % COUNTER_STRIDE) / WG_PER_IN * TILE_ROW)
#define WEI_COL_START   (grid_x / COUNTER_STRIDE)
#define WEI_WI_COL_START    (lid_x % PER_ITER_STRIDE)
#if STRIDE == 2
#define IN_COL_START_REAL   (((grid_x % WG_PER_IN) * TILE_COL) + (lid_x % TILE_COL))
#define IN_WI_COL_START_REAL    (IN_COL_START_REAL / OW * OW * 4 + IN_COL_START_REAL % OW * 2)
#define IN_COL_START    ((grid_x % WG_PER_IN) * TILE_COL)
#define IN_WI_COL_START (lid_x % TILE_COL)
#else
#define IN_COL_START    ((grid_x % WG_PER_IN) * TILE_COL)
#define IN_WI_COL_START (lid_x % TILE_COL)
#endif
#define OUT_WI_PER_ROW      (TILE_COL / PER_WI_TILE_COL)
#define OUT_WI_ROW_START    (lid_x / OUT_WI_PER_ROW * PER_WI_TILE_ROW)
#define OUT_WI_COL_START    ((lid_x % OUT_WI_PER_ROW) * PER_WI_TILE_COL)


#define OFFSET_METHOD 0
#if OFFSET_METHOD == 0
#define OFFSET  ((grid_x % COUNTER_STRIDE * STRIDE_WEI / 4096 * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 1
#define OFFSET  (((grid_x / COUNTER_STRIDE) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 2
#define OFFSET  (((grid_x % COUNTER_STRIDE) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 3
#define OFFSET  (((grid_x / WG_PER_WEI) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 4
#define OFFSET  (((grid_x % WG_PER_WEI) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 5
#define OFFSET  (((grid_x / WG_PER_IN) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 6
#define OFFSET  (((grid_x % WG_PER_IN) * PER_ITER_STRIDE) % split_stride)
#endif


#define COUNTER_INDEX   (grid_x % COUNTER_STRIDE)
#define LOCAL_STRIDE    (LOCAL_X / PER_ITER_STRIDE)

#define NOFILLED

//global: (LOCAL_X * WG_PER_IN * WG_PER_WEI * GLOBAL_SPLITU, 1, 1)

__attribute__((reqd_work_group_size(LOCAL_X, 1, 1)))
__kernel void conv1x1_act(
    __global const float* wei,
    __global const float* in,
#if BIAS == 1
    __constant float* bias,
#endif
    __global float* out,
    float slope) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[LDS_WEI_ROW * LDS_WEI_STRIDE];
    __local float shared_in[LDS_IN_ROW * LDS_IN_STRIDE];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in = (__local float*)shared_in;

    __global const float* pWei = (__global const float*)(wei + WEI_ROW_START * STRIDE_WEI +
                                 WEI_COL_START * QSTRIDE + WEI_WI_COL_START);
#if STRIDE == 2
    __global const float* pIn = (__global const float*)(in + (IN_WI_COL_START_REAL) + WEI_COL_START *
                                QSTRIDE * STRIDE_IN_REAL);
#else
    __global const float* pIn = (__global const float*)(in + (IN_COL_START + IN_WI_COL_START) +
                                WEI_COL_START * QSTRIDE * STRIDE_IN);
#endif
    __global float* pOut = (__global float*)(out + (IN_COL_START + OUT_WI_COL_START) +
                           (WEI_ROW_START + OUT_WI_ROW_START) * STRIDE_IN);

#if ATOMIC == 1
    __global uint* pBarrier = (__global uint*)(out + OUT_BATCH_STRIDE * N);
    __global uint* pCounter = (__global uint*)(out + OUT_BATCH_STRIDE * N + COUNTER_STRIDE);
#elif ATOMIC == 2
    volatile __global uint* pCounter = (volatile __global uint*)(out + OUT_BATCH_STRIDE * N);
#endif

#if BIAS == 1
    __constant float* pBias = (__constant float*)(bias + WEI_ROW_START + OUT_WI_ROW_START);
#endif

    uint split_stride;

    if ((GROUP_ITER + 1) * QSTRIDE < C) {
        split_stride = QSTRIDE;
    } else if ((GROUP_ITER + 1) * QSTRIDE - C < QSTRIDE) {
        split_stride = C - (GROUP_ITER) * QSTRIDE;
    } else {
        split_stride = 0;
    }

    ushort iter = split_stride / PER_ITER_STRIDE;

    float previous_value;
    uint prevVal;
    uint newVal;

#if ATOMIC == 1

    if (grid_x < COUNTER_STRIDE) {
        if (lid_x == 0) {
            *(pBarrier + COUNTER_INDEX) = 0;
            *(pCounter + COUNTER_INDEX) = 0;
        }
    }

#elif ATOMIC == 2

    if (grid_x < COUNTER_STRIDE) {
        if (lid_x == 0) {
            *(pCounter + COUNTER_INDEX) = 0;
        }
    }

#endif

    float sum[N][PER_WI_TILE_ROW][PER_WI_TILE_COL] = { { { 0.0f } } };

    uint offset = OFFSET;

#if BRANCH == 1

    for (ushort k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
#if 1
            shared_wei[WEI_WI_COL_START + (((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * LDS_WEI_STRIDE] =
                pWei[(((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * STRIDE_WEI + offset];
            prefetch(pWei + (((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
#else
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            prefetch(pWei + (LDS_WEI_ROW_START + i) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
#endif
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN_REAL -
                         IN_WI_COL_START_REAL + IN_COL_START_REAL / OW * OW * 4 + n * IN_BATCH_STRIDE, 64);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN -
                         IN_WI_COL_START + n * IN_BATCH_STRIDE, 32);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] += pShared_wei[j + (OUT_WI_ROW_START + m) * LDS_WEI_STRIDE] *
                                        pShared_in[j + (OUT_WI_COL_START + l) * LDS_IN_STRIDE];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 2

    for (uchar k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] += pShared_wei[j + (OUT_WI_ROW_START + m) * LDS_WEI_STRIDE] *
                                        pShared_in[j * LDS_IN_STRIDE + (OUT_WI_COL_START + l)];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 3

    for (uchar k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * LDS_WEI_STRIDE + (LDS_WEI_ROW_START + i) + (LDS_WEI_ROW_START + i) / 32]
                = pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            prefetch(pWei + (LDS_WEI_ROW_START + i) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN_REAL -
                         IN_WI_COL_START_REAL + IN_COL_START_REAL / OW * OW * 4 + n * IN_BATCH_STRIDE, 64);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN -
                         IN_WI_COL_START + n * IN_BATCH_STRIDE, 32);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar l = 0; l < PER_WI_TILE_COL; l++)
                    for (uchar m = 0; m < PER_WI_TILE_ROW; m++) {
                        sum[n][m][l] +=
                            pShared_wei[j * LDS_WEI_STRIDE + (OUT_WI_ROW_START + m) + (OUT_WI_ROW_START + m) / 32] *
                            pShared_in[j + (OUT_WI_COL_START + l) * LDS_IN_STRIDE];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 4

    for (uchar k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * LDS_WEI_STRIDE + (LDS_WEI_ROW_START + i) + (LDS_WEI_ROW_START + i) / 32]
                = pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] +=
                            pShared_wei[j * LDS_WEI_STRIDE + (OUT_WI_ROW_START + m) + (OUT_WI_ROW_START + m) / 32] *
                            pShared_in[j * LDS_IN_STRIDE + (OUT_WI_COL_START + l)];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#endif

#if ATOMIC == 0

    if (GROUP_ITER == 0) {
        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = sum[n][j][i] + pBias[j];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = sum[n][j][i];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#endif
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = sum[n][j][i] + pBias[j];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = sum[n][j][i];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#endif
        }
    }

#elif ATOMIC == 1

    if (GROUP_ITER == 0) {
        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        }

#ifdef NOFILLED
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            *(pBarrier + COUNTER_INDEX) = BARRIER_MARK;
        }

#endif
    } else {
#ifdef NOFILLED

        if (lid_x == 0) {
            do {
                newVal = BARRIER_MARK;
            } while (atomic_cmpxchg((__global uint*)(pBarrier + COUNTER_INDEX), BARRIER_MARK,
                                    newVal) != BARRIER_MARK);
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
#endif

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }

        if (GROUP_ITER == GSU_MINUS_ONE) {
            if (lid_x == 0) {
                do {
                    newVal = GSU_MINUS_ONE;
                } while (atomic_cmpxchg((__global uint*)(pCounter + COUNTER_INDEX), GSU_MINUS_ONE,
                                        newVal) != GSU_MINUS_ONE);
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

            if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
                for (uint n = 0; n < N; n++)
                    for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                        for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                            pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] *= (pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] >
                                    0.0f ? 1.0f : slope);
                        }
            } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
                for (uint n = 0; n < N; n++)
                    for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                        for (uint i = 0; i < 1; i++) {
                            pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] *= (pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] >
                                    0.0f ? 1.0f : slope);
                        }
            }
        }
    }

#elif ATOMIC == 2

    if (GROUP_ITER == 0) {
        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }
    } else if (GROUP_ITER == GSU_MINUS_ONE) {
#ifdef NOFILLED

        if (lid_x == 0) {
            do {
            } while (atomic_cmpxchg((volatile __global uint*)(pCounter + COUNTER_INDEX), GROUP_ITER,
                                    GROUP_ITER) < GROUP_ITER);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp * (temp > 0.0f ? 1.0f : slope);
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp * (temp > 0.0f ? 1.0f : slope);
                    }
        }
    } else {
        if (lid_x == 0) {
            do {
            } while (atomic_cmpxchg((volatile __global uint*)(pCounter + COUNTER_INDEX), GROUP_ITER,
                                    GROUP_ITER) < 1);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

#if 1

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        }

#else

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp;
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp;
                    }
        }

#endif
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }
    }

#endif
}

#elif KERNEL_METHOD == 2

#ifndef N
#define N   (1)
#endif
#ifndef C
#define C   (1024)
#endif
#ifndef H
#define H   (7)
#endif
#ifndef W
#define W   (7)
#endif
#ifndef K
#define K   (1024)
#endif

#define STRIDE_WEI  C
#define STRIDE_IN   (49)
#define QSTRIDE (C >> 2)
#define ITER    (QSTRIDE >> 5)
#define LDS_WEI_STRIDE  (17)
#define LDS_IN_STRIDE   (132)

#define IN_BATCH_STRIDE     (C * H * W)
#define OUT_BATCH_STRIDE    (K)

void reduce(__local float* buffer, int tid) {
    if ((tid & 63) < 32) {
        buffer[tid + (tid >> 5)] += buffer[tid + 32 + ((tid + 32) >> 5)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((tid & 63) < 16) {
        buffer[tid + (tid >> 5)] += buffer[tid + 16 + ((tid + 16) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 8 + ((tid + 8) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 4 + ((tid + 4) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 2 + ((tid + 2) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 1 + ((tid + 1) >> 5)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void conv1x1_act_pool(
    __global const float* wei,
    __global const float* in,
#ifdef BIAS
    __constant float* bias,
#endif
    __global float* out,
    float slope) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[1024 * 4];
    __local float shared_in[2560 * 4];
    __local float shared_result[512 * 4];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in = (__local float*)shared_in;

    __global const float* pWei = (__global const float*)(wei + ((grid_x & 63) << 4) * STRIDE_WEI +
                                 (lid_x & 127));
    __global const float* pIn = (__global const float*)(in + ((lid_x & 63)));
    __global float* pOut = (__global float*)(out + ((grid_x & 63) << 4) + (lid_x >> 6));
#ifdef BIAS
    __constant float* pBias = (__constant float*)(bias + ((grid_x & 63) << 4) + (lid_x >> 6));
#endif


    float sum[N] = { 0.0f };

    uint offset = ((grid_x & 63) << 7) % STRIDE_WEI;


    for (uchar k = 0; k < ITER; k++, offset = (offset + 128) % STRIDE_WEI) {
        for (uchar i = 0; i < 2; i++) {
            shared_wei[(lid_x & 127) + ((lid_x & 127) >> 5) + ((lid_x >> 7 << 1) + i) * LDS_IN_STRIDE] =
                pWei[((lid_x >> 7 << 1) + i) * STRIDE_WEI + offset];
        }

        for (uchar n = 0; n < N; n++) {
            for (uchar i = 0; i < 8; i++) {
                shared_in[((lid_x >> 6 << 3) + i) * 66 + (lid_x & 63) + ((lid_x & 63) >> 5)] = ((
                            lid_x & 63) < STRIDE_IN ? pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] :
                        0.0f); //pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < 128; j++) {
                sum[n] += pShared_wei[j + (j >> 5) + ((lid_x >> 6) * LDS_IN_STRIDE)] *
                          pShared_in[j * 66 + (lid_x & 63) + ((lid_x & 63) >> 5)];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }


    for (uint n = 0; n < N; n++) {
#ifdef BIAS
        sum[n] += (lid_x & 63) < STRIDE_IN ? pBias[0] : 0.0f;
#endif
        sum[n] *= (sum[n] > 0.0f ? 1.0f : slope);
        shared_result[lid_x + (lid_x >> 5)] = sum[n];
        barrier(CLK_LOCAL_MEM_FENCE);

        reduce(shared_result, lid_x);

        if ((lid_x & 63) == 0) {
            pOut[n * OUT_BATCH_STRIDE] = shared_result[(lid_x) + (lid_x >> 5)] / 49.0f;
        }
    }
}

#elif KERNEL_METHOD == 3

#ifndef N
#define N   (1)
#endif
#ifndef C
#define C   (320)
#endif
#ifndef H
#define H   (7)
#endif
#ifndef W
#define W   (7)
#endif
#ifndef K
#define K   (1280)
#endif

#define PER_ITER_STRIDE (80)
#define STRIDE_WEI  C
#define STRIDE_IN   (H * W)
#define ITER    (STRIDE_WEI / PER_ITER_STRIDE)

#define LDS_WEI_STRIDE  (132)
#define LDS_WEI_COL (128)
#define LDS_WEI_COL_REAL    (80)
#define LDS_WEI_ROW (20)
#define LDS_IN_STRIDE   (66)
#define LDS_IN_COL_REAL (STRIDE_IN)
#define LDS_IN_COL  (64)
#define LDS_IN_ROW  (80)
#define LDS_WEI_READ_ITER   (3)
#define LDS_IN_READ_ITER    (5)
#define PER_WI_TILE_ROW (1)
#define PER_WI_TILE_COL (1)

#define COUNTER_STRIDE  64
#define IN_COL_WG   1
#define GROUP_ITER  (grid_x / COUNTER_STRIDE)
#define WEI_ROW_START   ((grid_x % COUNTER_STRIDE) / IN_COL_WG * LDS_WEI_ROW)
#define WEI_COL_START   (grid_x / COUNTER_STRIDE)
#define WEI_WI_COL_START    (lid_x % LDS_WEI_COL)
#define IN_COL_START    ((grid_x % IN_COL_WG) * LDS_IN_ROW)
#define IN_WI_COL_START (lid_x % COUNTER_STRIDE)
#define OUT_WI_ROW_START    (lid_x / STRIDE_IN)
#define OUT_WI_COL_START    (lid_x & 15)
#define OUT_PER_WG  (STRIDE_IN * LDS_WEI_ROW)
#define OUT_PER_WG_REAL (LDS_WEI_ROW)
#define OUT_WI_INDEX    (lid_x)
#define LDS_WEI_WI  (lid_x % LDS_WEI_COL)
#define LDS_WEI_ROW_START   ((lid_x / LDS_WEI_COL))
#define LDS_IN_ROW_START    ((lid_x >> 6) * LDS_IN_READ_ITER)
#define COMPUTE_WEI_WI_INDEX    (lid_x / STRIDE_IN)
#define COMPUTE_IN_WI_INDEX (lid_x % STRIDE_IN)
#define OFFSET  ((grid_x % COUNTER_STRIDE) * PER_ITER_STRIDE % STRIDE_WEI)

#define IN_BATCH_STRIDE     (C * H * W)
#define OUT_BATCH_STRIDE    (K)

void reduce(__local float* buffer, uint tid, uint start, uint upper) {
    tid += start;

    if (tid < upper) {
        if ((tid & 63) < 32) {
            buffer[tid + (tid >> 5)] += buffer[tid + 32 + ((tid + 32) >> 5)];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < upper) {
        if ((tid & 63) < 16) {
            buffer[tid + (tid >> 5)] += buffer[tid + 16 + ((tid + 16) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 8 + ((tid + 8) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 4 + ((tid + 4) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 2 + ((tid + 2) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 1 + ((tid + 1) >> 5)];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void conv1x1_act_pool(
    __global const float* wei,
    __global const float* in,
#ifdef BIAS
    __constant float* bias,
#endif
    __global float* out,
    float slope) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[1024 * 4];
    __local float shared_in[2560 * 4];
    __local float shared_result[512 * 4];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in = (__local float*)shared_in;

    __global const float* pWei = (__global const float*)(wei + (WEI_ROW_START * STRIDE_WEI +
                                 WEI_WI_COL_START));//
    __global const float* pIn = (__global const float*)(in + (IN_WI_COL_START));//
    __global float* pOut = (__global float*)(out + (WEI_ROW_START + lid_x));//
#ifdef BIAS
    __constant float* pBias = (__constant float*)(bias + (WEI_ROW_START + OUT_WI_ROW_START));//
#endif


    float sum[N] = { 0.0f };

    uint offset = OFFSET;

    for (uint i = 0; i < 2; i++) {
        shared_result[lid_x + i * 1024] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % STRIDE_WEI) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            if (LDS_WEI_ROW_START + i * 8 < LDS_WEI_ROW) {
                shared_wei[WEI_WI_COL_START + (WEI_WI_COL_START >> 5) + (LDS_WEI_ROW_START + i * 8) * LDS_WEI_STRIDE]
                    = (LDS_WEI_WI < LDS_WEI_COL_REAL ? pWei[(LDS_WEI_ROW_START + i * 8) * STRIDE_WEI + offset] : 0.0f);
            }
        }

        for (uchar n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START + (IN_WI_COL_START >> 5)] =
                    (IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] :
                     0.0f); //pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                sum[n] += (lid_x < OUT_PER_WG ?
                           pShared_wei[j + (j >> 5) + (COMPUTE_WEI_WI_INDEX * LDS_WEI_STRIDE)] *
                           pShared_in[j * LDS_IN_STRIDE + COMPUTE_IN_WI_INDEX + (COMPUTE_IN_WI_INDEX >> 5)] : 0.0f);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }


    for (uint n = 0; n < N; n++) {
#ifdef BIAS
        sum[n] += lid_x < OUT_PER_WG ? pBias[0] : 0.0f;
#endif
        sum[n] *= (sum[n] > 0.0f ? 1.0f : slope);
        shared_result[(COMPUTE_WEI_WI_INDEX * 64 + COMPUTE_IN_WI_INDEX) + ((COMPUTE_WEI_WI_INDEX * 64 + COMPUTE_IN_WI_INDEX) >> 5)]
            = sum[n];
        barrier(CLK_LOCAL_MEM_FENCE);

        reduce(shared_result, lid_x, 0, K);
        reduce(shared_result, lid_x, 1024, K);

        if (lid_x < OUT_PER_WG_REAL) {
            pOut[n * OUT_BATCH_STRIDE] = shared_result[(lid_x << 6) + (lid_x << 6 >> 5)] / 49.0f;
        }
    }

}

#elif KERNEL_METHOD == 4
#if GLOBAL_SPLITU == 1
#define ATOMIC  0
#else
#define ATOMIC  2
#endif

///////////////////////////////////////////
#ifndef LOCAL_X
#define LOCAL_X     256
#endif

#ifndef METHOD
#define METHOD  1
#endif

#ifndef BRANCH
#if N == 1
#define BRANCH  1
#elif N == 2
#define BRANCH  3
#endif
#endif

#define GSU_MINUS_ONE   (GLOBAL_SPLITU - 1)
#define BARRIER_MARK    (0xffffffff)
#define STRIDE_WEI  (C)
#define ALIGNED_C   (C + GLOBAL_SPLITU * PER_ITER_STRIDE - 1) / (GLOBAL_SPLITU * PER_ITER_STRIDE) * (GLOBAL_SPLITU * PER_ITER_STRIDE)
#if STRIDE == 2
#define OW  (W / 2)
#define OH  (H / 2)
#define STRIDE_IN_REAL  (H * W)
#else
#define OW  (W)
#define OH  (H)
#endif
#define STRIDE_IN   (OH * OW)


#define WG_PER_IN   ((STRIDE_IN + TILE_COL - 1) / TILE_COL)
#define WG_PER_WEI  ((K + TILE_ROW - 1) / TILE_ROW)

#define QSTRIDE (ALIGNED_C / GLOBAL_SPLITU)
#define ITER    (QSTRIDE / PER_ITER_STRIDE)

#if BRANCH == 1 || BRANCH == 2
#define LDS_WEI_COL (PER_ITER_STRIDE)
#define LDS_WEI_STRIDE  (LDS_WEI_COL + (LDS_WEI_COL + 31) / 32)
#define LDS_WEI_ROW (TILE_ROW)
#define WEI_READ_LINE   LDS_WEI_COL
#elif BRANCH == 3 || BRANCH == 4
#define LDS_WEI_COL (TILE_ROW)
#define LDS_WEI_STRIDE  (LDS_WEI_COL + (LDS_WEI_COL + 31) / 32)
#define LDS_WEI_ROW (PER_ITER_STRIDE)
#define WEI_READ_LINE   LDS_WEI_ROW
#endif
#if BRANCH == 1 || BRANCH == 3
#define LDS_IN_COL  (PER_ITER_STRIDE)
#define LDS_IN_STRIDE   (LDS_IN_COL + (LDS_IN_COL + 31) / 32)
#define LDS_IN_ROW  (TILE_COL)
#define IN_READ_LINE    LDS_IN_ROW
#elif BRANCH == 2 || BRANCH == 4
#define LDS_IN_COL  (TILE_COL)
#define LDS_IN_STRIDE   (LDS_IN_COL + (LDS_IN_COL + 31) / 32)
#define LDS_IN_ROW  (PER_ITER_STRIDE)
#define IN_READ_LINE    LDS_IN_COL
#endif

#define LDS_WEI_READ_ITER   (LDS_WEI_COL * LDS_WEI_ROW / LOCAL_X)
#define LDS_IN_READ_ITER    (LDS_IN_COL * LDS_IN_ROW / LOCAL_X)
#define LDS_WEI_ROW_START   ((lid_x / WEI_READ_LINE) * LDS_WEI_READ_ITER)
#define LDS_IN_ROW_START    ((lid_x / IN_READ_LINE) * LDS_IN_READ_ITER)

#define IN_BATCH_STRIDE     (C * H * W)
#define OUT_BATCH_STRIDE    (K * OH * OW)

#define COUNTER_STRIDE  (WG_PER_IN * WG_PER_WEI)
#define GROUP_ITER  (grid_x / COUNTER_STRIDE)
#define WEI_ROW_START   ((grid_x % COUNTER_STRIDE) / WG_PER_IN * TILE_ROW)
#define WEI_COL_START   (grid_x / COUNTER_STRIDE)
#define WEI_WI_COL_START    (lid_x % PER_ITER_STRIDE)
#if STRIDE == 2
#define IN_COL_START_REAL   (((grid_x % WG_PER_IN) * TILE_COL) + (lid_x % TILE_COL))
#define IN_WI_COL_START_REAL    (IN_COL_START_REAL / OW * OW * 4 + IN_COL_START_REAL % OW * 2)
#define IN_COL_START    ((grid_x % WG_PER_IN) * TILE_COL)
#define IN_WI_COL_START (lid_x % TILE_COL)
#else
#define IN_COL_START    ((grid_x % WG_PER_IN) * TILE_COL)
#define IN_WI_COL_START (lid_x % TILE_COL)
#endif
#define OUT_WI_PER_ROW      (TILE_COL / PER_WI_TILE_COL)
#define OUT_WI_ROW_START    (lid_x / OUT_WI_PER_ROW * PER_WI_TILE_ROW)
#define OUT_WI_COL_START    ((lid_x % OUT_WI_PER_ROW) * PER_WI_TILE_COL)



#define OFFSET_METHOD 0
#if OFFSET_METHOD == 0
#define OFFSET  ((grid_x % COUNTER_STRIDE * STRIDE_WEI / 4096 * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 1
#define OFFSET  (((grid_x / COUNTER_STRIDE) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 2
#define OFFSET  (((grid_x % COUNTER_STRIDE) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 3
#define OFFSET  (((grid_x / WG_PER_WEI) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 4
#define OFFSET  (((grid_x % WG_PER_WEI) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 5
#define OFFSET  (((grid_x / WG_PER_IN) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 6
#define OFFSET  (((grid_x % WG_PER_IN) * PER_ITER_STRIDE) % split_stride)
#endif


#define COUNTER_INDEX   (grid_x % COUNTER_STRIDE)
#define LOCAL_STRIDE    (LOCAL_X / PER_ITER_STRIDE)

#define NOFILLED

//global: (LOCAL_X * WG_PER_IN * WG_PER_WEI * GLOBAL_SPLITU, 1, 1)

__attribute__((reqd_work_group_size(LOCAL_X, 1, 1)))
__kernel void conv1x1_act(
    __global const float* wei,
    __global const float* in,
#if BIAS == 1
    __constant float* bias,
#endif
    __global float* out,
    float slope) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[LDS_WEI_ROW * LDS_WEI_STRIDE];
    __local float shared_in[LDS_IN_ROW * LDS_IN_STRIDE];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in = (__local float*)shared_in;

    __global const float* pWei = (__global const float*)(wei + WEI_ROW_START * STRIDE_WEI +
                                 WEI_COL_START * QSTRIDE + WEI_WI_COL_START);//
#if STRIDE == 2
    __global const float* pIn = (__global const float*)(in + (IN_WI_COL_START_REAL) + WEI_COL_START *
                                QSTRIDE * STRIDE_IN_REAL);//
#else
    __global const float* pIn = (__global const float*)(in + (IN_COL_START + IN_WI_COL_START) +
                                WEI_COL_START * QSTRIDE * STRIDE_IN);//
#endif
    __global float* pOut = (__global float*)(out + (IN_COL_START + OUT_WI_COL_START) +
                           (WEI_ROW_START + OUT_WI_ROW_START) * STRIDE_IN);//

#if ATOMIC == 2
    volatile __global uint* pCounter = (volatile __global uint*)(out + OUT_BATCH_STRIDE * N);//
#endif

#if BIAS == 1
    __constant float* pBias = (__constant float*)(bias + WEI_ROW_START + OUT_WI_ROW_START);//
#endif

    uint split_stride;

    if ((GROUP_ITER + 1) * QSTRIDE < C) {
        split_stride = QSTRIDE;
    } else if ((GROUP_ITER + 1) * QSTRIDE - C < QSTRIDE) {
        split_stride = C - (GROUP_ITER) * QSTRIDE;
    } else {
        split_stride = 0;
    }

    uint iter = (split_stride + PER_ITER_STRIDE - 1) / PER_ITER_STRIDE;

    uint remainder = C % PER_ITER_STRIDE;
    remainder = (remainder == 0 ? PER_ITER_STRIDE : remainder);
    uint per_iter_stride;

    float previous_value;
    uint prevVal;
    uint newVal;

#if ATOMIC == 2

    if (grid_x < COUNTER_STRIDE) {
        if (lid_x == 0) {
            *(pCounter + COUNTER_INDEX) = 0;
        }
    }

#endif

    float sum[N][PER_WI_TILE_ROW][PER_WI_TILE_COL] = { { { 0.0f } } };

    uint offset = 0;//OFFSET;

#if BRANCH == 1

    for (uint k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        per_iter_stride = ((k == iter - 1 && GROUP_ITER == GSU_MINUS_ONE) ? remainder : PER_ITER_STRIDE);

        for (uint i = 0; i < LDS_WEI_READ_ITER; i++) {
#if 1
            shared_wei[WEI_WI_COL_START + (((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * LDS_WEI_STRIDE] =
                pWei[(((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * STRIDE_WEI + offset];
            prefetch(pWei + (((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
#else
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            prefetch(pWei + (LDS_WEI_ROW_START + i) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
#endif
        }

        for (uint n = 0; n < N; n++) {
            for (uint i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN_REAL -
                         IN_WI_COL_START_REAL + IN_COL_START_REAL / OW * OW * 4 + n * IN_BATCH_STRIDE, 64);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN -
                         IN_WI_COL_START + n * IN_BATCH_STRIDE, 32);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint j = 0; j < per_iter_stride; j++) {
                for (uint m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uint l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] += pShared_wei[j + (OUT_WI_ROW_START + m) * LDS_WEI_STRIDE] *
                                        pShared_in[j + (OUT_WI_COL_START + l) * LDS_IN_STRIDE];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 2

    for (uint k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uint i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uint n = 0; n < N; n++) {
            for (uint i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint j = 0; j < PER_ITER_STRIDE; j++) {
                for (uint m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uint l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] += pShared_wei[j + (OUT_WI_ROW_START + m) * LDS_WEI_STRIDE] *
                                        pShared_in[j * LDS_IN_STRIDE + (OUT_WI_COL_START + l)];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 3

    for (uint k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uint i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * LDS_WEI_STRIDE + (LDS_WEI_ROW_START + i) + (LDS_WEI_ROW_START + i) / 32]
                = pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            prefetch(pWei + (LDS_WEI_ROW_START + i) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
        }

        for (uint n = 0; n < N; n++) {
            for (uint i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN_REAL -
                         IN_WI_COL_START_REAL + IN_COL_START_REAL / OW * OW * 4 + n * IN_BATCH_STRIDE, 64);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN -
                         IN_WI_COL_START + n * IN_BATCH_STRIDE, 32);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint j = 0; j < PER_ITER_STRIDE; j++) {
                for (uint l = 0; l < PER_WI_TILE_COL; l++)
                    for (uint m = 0; m < PER_WI_TILE_ROW; m++) {
                        sum[n][m][l] +=
                            pShared_wei[j * LDS_WEI_STRIDE + (OUT_WI_ROW_START + m) + (OUT_WI_ROW_START + m) / 32] *
                            pShared_in[j + (OUT_WI_COL_START + l) * LDS_IN_STRIDE];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 4

    for (uint k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uint i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * LDS_WEI_STRIDE + (LDS_WEI_ROW_START + i) + (LDS_WEI_ROW_START + i) / 32]
                = pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uint n = 0; n < N; n++) {
            for (uint i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint j = 0; j < PER_ITER_STRIDE; j++) {
                for (uint m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uint l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] +=
                            pShared_wei[j * LDS_WEI_STRIDE + (OUT_WI_ROW_START + m) + (OUT_WI_ROW_START + m) / 32] *
                            pShared_in[j * LDS_IN_STRIDE + (OUT_WI_COL_START + l)];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#endif

    uint out_width = STRIDE_IN;
    uint out_height = K;
    uint mod_width = out_width % PER_WI_TILE_COL;
    uint mod_height = out_height % PER_WI_TILE_ROW;
    uint per_wi_tile_col = 0;
    uint per_wi_tile_row = 0;

    if (mod_width == 0) {
        if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            per_wi_tile_col = PER_WI_TILE_COL;
        }
    } else {
        if (IN_COL_START + OUT_WI_COL_START + mod_width < STRIDE_IN) {
            per_wi_tile_col = PER_WI_TILE_COL;
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            per_wi_tile_col = mod_width;
        }
    }

    if (mod_height == 0) {
        if (WEI_ROW_START + OUT_WI_ROW_START < out_height) {
            per_wi_tile_row = PER_WI_TILE_ROW;
        }
    } else {
        if (WEI_ROW_START + OUT_WI_ROW_START + mod_height < out_height) {
            per_wi_tile_row = PER_WI_TILE_ROW;
        } else if (WEI_ROW_START + OUT_WI_ROW_START < out_height) {
            per_wi_tile_row = mod_height;
        }
    }

#if ATOMIC == 0

    for (uint n = 0; n < N; n++)
        for (uint j = 0; j < per_wi_tile_row; j++)
            for (uint i = 0; i < per_wi_tile_col; i++) {
#if BIAS == 1
                float temp = sum[n][j][i] + pBias[j];
#else
                float temp = sum[n][j][i];
#endif
                temp *= (temp > 0.0f ? 1.0f : slope);
                pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
            }

#elif ATOMIC == 2

    if (GROUP_ITER == 0) {
        for (uint n = 0; n < N; n++)
            for (uint j = 0; j < per_wi_tile_row; j++)
                for (uint i = 0; i < per_wi_tile_col; i++) {
#if BIAS == 1
                    *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
#else
                    *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
#endif
                }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }
    } else if (GROUP_ITER == GSU_MINUS_ONE) {
#ifdef NOFILLED

        if (lid_x == 0) {
            do {
            } while (atomic_cmpxchg((volatile __global uint*)(pCounter + COUNTER_INDEX), GROUP_ITER,
                                    GROUP_ITER) < GROUP_ITER);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        for (uint n = 0; n < N; n++)
            for (uint j = 0; j < per_wi_tile_row; j++)
                for (uint i = 0; i < per_wi_tile_col; i++) {
                    float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                    temp += sum[n][j][i];
                    *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp * (temp > 0.0f ? 1.0f : slope);
                }
    } else {
        if (lid_x == 0) {
            do {
            } while (atomic_cmpxchg((volatile __global uint*)(pCounter + COUNTER_INDEX), GROUP_ITER,
                                    GROUP_ITER) < 1);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint n = 0; n < N; n++)
            for (uint j = 0; j < per_wi_tile_row; j++)
                for (uint i = 0; i < per_wi_tile_col; i++) {
                    do {
                        previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        prevVal = as_uint(previous_value);
                        newVal = as_uint(sum[n][j][i] + previous_value);
                    } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                            newVal) != prevVal);
                }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }
    }

#endif
}

#elif KERNEL_METHOD == 5
#elif KERNEL_METHOD == 6
#elif KERNEL_METHOD == 7
#elif KERNEL_METHOD == 8
#elif KERNEL_METHOD == 9
#endif

#else // #ifndef MACRO

#ifndef KERNEL_METHOD
#define KERNEL_METHOD   1
#endif


#if KERNEL_METHOD == 1

#if GLOBAL_SPLITU == 1
#define ATOMIC  0
#else
#define ATOMIC  2
#endif

///////////////////////////////////////////
#ifndef LOCAL_X
#define LOCAL_X     256
#endif

#ifndef METHOD
#define METHOD  1
#endif

#ifndef BRANCH
#if N == 1
#define BRANCH  1
#elif N == 2
#define BRANCH  3
#endif
#endif

#define GSU_MINUS_ONE   (GLOBAL_SPLITU - 1)
#define BARRIER_MARK    (0xffffffff)
#define STRIDE_WEI  (C)
#define ALIGNED_C   (C + GLOBAL_SPLITU * PER_ITER_STRIDE - 1) / (GLOBAL_SPLITU * PER_ITER_STRIDE) * (GLOBAL_SPLITU * PER_ITER_STRIDE)
#if STRIDE == 2
#define OW  (W / 2)
#define OH  (H / 2)
#define STRIDE_IN_REAL  (H * W)
#else
#define OW  (W)
#define OH  (H)
#endif
#define STRIDE_IN   (OH * OW)


#define WG_PER_IN   ((STRIDE_IN + TILE_COL - 1) / TILE_COL)
#define WG_PER_WEI  ((K + TILE_ROW - 1) / TILE_ROW)

#define QSTRIDE (ALIGNED_C / GLOBAL_SPLITU)
#define ITER    (QSTRIDE / PER_ITER_STRIDE)

#if BRANCH == 1 || BRANCH == 2
#define LDS_WEI_COL (PER_ITER_STRIDE)
#define LDS_WEI_STRIDE  (LDS_WEI_COL + (LDS_WEI_COL + 31) / 32)
#define LDS_WEI_ROW (TILE_ROW)
#define WEI_READ_LINE   LDS_WEI_COL
#elif BRANCH == 3 || BRANCH == 4
#define LDS_WEI_COL (TILE_ROW)
#define LDS_WEI_STRIDE  (LDS_WEI_COL + (LDS_WEI_COL + 31) / 32)
#define LDS_WEI_ROW (PER_ITER_STRIDE)
#define WEI_READ_LINE   LDS_WEI_ROW
#endif
#if BRANCH == 1 || BRANCH == 3
#define LDS_IN_COL  (PER_ITER_STRIDE)
#define LDS_IN_STRIDE   (LDS_IN_COL + (LDS_IN_COL + 31) / 32)
#define LDS_IN_ROW  (TILE_COL)
#define IN_READ_LINE    LDS_IN_ROW
#elif BRANCH == 2 || BRANCH == 4
#define LDS_IN_COL  (TILE_COL)
#define LDS_IN_STRIDE   (LDS_IN_COL + (LDS_IN_COL + 31) / 32)
#define LDS_IN_ROW  (PER_ITER_STRIDE)
#define IN_READ_LINE    LDS_IN_COL
#endif

#define LDS_WEI_READ_ITER   (LDS_WEI_COL * LDS_WEI_ROW / LOCAL_X)
#define LDS_IN_READ_ITER    (LDS_IN_COL * LDS_IN_ROW / LOCAL_X)
#define LDS_WEI_ROW_START   ((lid_x / WEI_READ_LINE) * LDS_WEI_READ_ITER)
#define LDS_IN_ROW_START    ((lid_x / IN_READ_LINE) * LDS_IN_READ_ITER)

#define IN_BATCH_STRIDE     (C * H * W)
#define OUT_BATCH_STRIDE    (K * OH * OW)

#define COUNTER_STRIDE  (WG_PER_IN * WG_PER_WEI)
#define GROUP_ITER  (grid_x / COUNTER_STRIDE)
#define WEI_ROW_START   ((grid_x % COUNTER_STRIDE) / WG_PER_IN * TILE_ROW)
#define WEI_COL_START   (grid_x / COUNTER_STRIDE)
#define WEI_WI_COL_START    (lid_x % PER_ITER_STRIDE)
#if STRIDE == 2
#define IN_COL_START_REAL   (((grid_x % WG_PER_IN) * TILE_COL) + (lid_x % TILE_COL))
#define IN_WI_COL_START_REAL    (IN_COL_START_REAL / OW * OW * 4 + IN_COL_START_REAL % OW * 2)
#define IN_COL_START    ((grid_x % WG_PER_IN) * TILE_COL)
#define IN_WI_COL_START (lid_x % TILE_COL)
#else
#define IN_COL_START    ((grid_x % WG_PER_IN) * TILE_COL)
#define IN_WI_COL_START (lid_x % TILE_COL)
#endif
#define OUT_WI_PER_ROW      (TILE_COL / PER_WI_TILE_COL)
#define OUT_WI_ROW_START    (lid_x / OUT_WI_PER_ROW * PER_WI_TILE_ROW)
#define OUT_WI_COL_START    ((lid_x % OUT_WI_PER_ROW) * PER_WI_TILE_COL)


#define OFFSET_METHOD 0
#if OFFSET_METHOD == 0
#define OFFSET  ((grid_x % COUNTER_STRIDE * STRIDE_WEI / 4096 * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 1
#define OFFSET  (((grid_x / COUNTER_STRIDE) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 2
#define OFFSET  (((grid_x % COUNTER_STRIDE) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 3
#define OFFSET  (((grid_x / WG_PER_WEI) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 4
#define OFFSET  (((grid_x % WG_PER_WEI) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 5
#define OFFSET  (((grid_x / WG_PER_IN) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 6
#define OFFSET  (((grid_x % WG_PER_IN) * PER_ITER_STRIDE) % split_stride)
#endif


#define COUNTER_INDEX   (grid_x % COUNTER_STRIDE)
#define LOCAL_STRIDE    (LOCAL_X / PER_ITER_STRIDE)

#define NOFILLED

//global: (LOCAL_X * WG_PER_IN * WG_PER_WEI * GLOBAL_SPLITU, 1, 1)

__attribute__((reqd_work_group_size(LOCAL_X, 1, 1)))
__kernel void conv1x1_act(
    __global const float* wei,
    __global const float* in,
#if BIAS == 1
    __constant float* bias,
#endif
    __global float* out,
    float slope, uint C, uint H, uint W, uint K) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[LDS_WEI_ROW * LDS_WEI_STRIDE];
    __local float shared_in[LDS_IN_ROW * LDS_IN_STRIDE];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in = (__local float*)shared_in;

    __global const float* pWei = (__global const float*)(wei + WEI_ROW_START * STRIDE_WEI +
                                 WEI_COL_START * QSTRIDE + WEI_WI_COL_START);//
#if STRIDE == 2
    __global const float* pIn = (__global const float*)(in + (IN_WI_COL_START_REAL) + WEI_COL_START *
                                QSTRIDE * STRIDE_IN_REAL);//
#else
    __global const float* pIn = (__global const float*)(in + (IN_COL_START + IN_WI_COL_START) +
                                WEI_COL_START * QSTRIDE * STRIDE_IN);//
#endif
    __global float* pOut = (__global float*)(out + (IN_COL_START + OUT_WI_COL_START) +
                           (WEI_ROW_START + OUT_WI_ROW_START) * STRIDE_IN);//

#if ATOMIC == 1
    __global uint* pBarrier = (__global uint*)(out + OUT_BATCH_STRIDE * N);//
    __global uint* pCounter = (__global uint*)(out + OUT_BATCH_STRIDE * N + COUNTER_STRIDE);//
#elif ATOMIC == 2
    volatile __global uint* pCounter = (volatile __global uint*)(out + OUT_BATCH_STRIDE * N);//
#endif

#if BIAS == 1
    __constant float* pBias = (__constant float*)(bias + WEI_ROW_START + OUT_WI_ROW_START);//
#endif

    uint split_stride;

    if ((GROUP_ITER + 1) * QSTRIDE < C) {
        split_stride = QSTRIDE;
    } else if ((GROUP_ITER + 1) * QSTRIDE - C < QSTRIDE) {
        split_stride = C - (GROUP_ITER) * QSTRIDE;
    } else {
        split_stride = 0;
    }

    ushort iter = split_stride / PER_ITER_STRIDE;

    float previous_value;
    uint prevVal;
    uint newVal;

#if ATOMIC == 1

    if (grid_x < COUNTER_STRIDE) {
        if (lid_x == 0) {
            *(pBarrier + COUNTER_INDEX) = 0;
            *(pCounter + COUNTER_INDEX) = 0;
        }
    }

    //barrier(CLK_GLOBAL_MEM_FENCE);
#elif ATOMIC == 2

    if (grid_x < COUNTER_STRIDE) {
        if (lid_x == 0) {
            *(pCounter + COUNTER_INDEX) = 0;
        }
    }

    //barrier(CLK_GLOBAL_MEM_FENCE);
#endif

    float sum[N][PER_WI_TILE_ROW][PER_WI_TILE_COL] = { { { 0.0f } } };

    uint offset = OFFSET;

#if BRANCH == 1

    for (ushort k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
#if 1
            shared_wei[WEI_WI_COL_START + (((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * LDS_WEI_STRIDE] =
                pWei[(((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * STRIDE_WEI + offset];
            prefetch(pWei + (((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
#else
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            prefetch(pWei + (LDS_WEI_ROW_START + i) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
#endif
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN_REAL -
                         IN_WI_COL_START_REAL + IN_COL_START_REAL / OW * OW * 4 + n * IN_BATCH_STRIDE, 64);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN -
                         IN_WI_COL_START + n * IN_BATCH_STRIDE, 32);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] += pShared_wei[j + (OUT_WI_ROW_START + m) * LDS_WEI_STRIDE] *
                                        pShared_in[j + (OUT_WI_COL_START + l) * LDS_IN_STRIDE];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 2

    for (uchar k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] += pShared_wei[j + (OUT_WI_ROW_START + m) * LDS_WEI_STRIDE] *
                                        pShared_in[j * LDS_IN_STRIDE + (OUT_WI_COL_START + l)];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 3

    for (uchar k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * LDS_WEI_STRIDE + (LDS_WEI_ROW_START + i) + (LDS_WEI_ROW_START + i) / 32]
                = pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            prefetch(pWei + (LDS_WEI_ROW_START + i) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN_REAL -
                         IN_WI_COL_START_REAL + IN_COL_START_REAL / OW * OW * 4 + n * IN_BATCH_STRIDE, 64);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN -
                         IN_WI_COL_START + n * IN_BATCH_STRIDE, 32);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar l = 0; l < PER_WI_TILE_COL; l++)
                    for (uchar m = 0; m < PER_WI_TILE_ROW; m++) {
                        sum[n][m][l] +=
                            pShared_wei[j * LDS_WEI_STRIDE + (OUT_WI_ROW_START + m) + (OUT_WI_ROW_START + m) / 32] *
                            pShared_in[j + (OUT_WI_COL_START + l) * LDS_IN_STRIDE];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 4

    for (uchar k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * LDS_WEI_STRIDE + (LDS_WEI_ROW_START + i) + (LDS_WEI_ROW_START + i) / 32]
                = pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] +=
                            pShared_wei[j * LDS_WEI_STRIDE + (OUT_WI_ROW_START + m) + (OUT_WI_ROW_START + m) / 32] *
                            pShared_in[j * LDS_IN_STRIDE + (OUT_WI_COL_START + l)];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#endif

#if ATOMIC == 0

    if (GROUP_ITER == 0) {
        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = sum[n][j][i] + pBias[j];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = sum[n][j][i];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#endif
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = sum[n][j][i] + pBias[j];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = sum[n][j][i];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#endif
        }
    }

#elif ATOMIC == 1

    if (GROUP_ITER == 0) {
        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        }

#ifdef NOFILLED
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            *(pBarrier + COUNTER_INDEX) = BARRIER_MARK;
        }

#endif
    } else {
#ifdef NOFILLED

        if (lid_x == 0) {
            do {
                newVal = BARRIER_MARK;
            } while (atomic_cmpxchg((__global uint*)(pBarrier + COUNTER_INDEX), BARRIER_MARK,
                                    newVal) != BARRIER_MARK);
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
#endif

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }

        if (GROUP_ITER == GSU_MINUS_ONE) {
            if (lid_x == 0) {
                do {
                    newVal = GSU_MINUS_ONE;
                } while (atomic_cmpxchg((__global uint*)(pCounter + COUNTER_INDEX), GSU_MINUS_ONE,
                                        newVal) != GSU_MINUS_ONE);
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

            if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
                for (uint n = 0; n < N; n++)
                    for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                        for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                            pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] *= (pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] >
                                    0.0f ? 1.0f : slope);
                        }
            } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
                for (uint n = 0; n < N; n++)
                    for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                        for (uint i = 0; i < 1; i++) {
                            pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] *= (pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] >
                                    0.0f ? 1.0f : slope);
                        }
            }
        }
    }

#elif ATOMIC == 2

    if (GROUP_ITER == 0) {
        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }
    } else if (GROUP_ITER == GSU_MINUS_ONE) {
#ifdef NOFILLED

        if (lid_x == 0) {
            do {
            } while (atomic_cmpxchg((volatile __global uint*)(pCounter + COUNTER_INDEX), GROUP_ITER,
                                    GROUP_ITER) < GROUP_ITER);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp * (temp > 0.0f ? 1.0f : slope);
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp * (temp > 0.0f ? 1.0f : slope);
                    }
        }
    } else {
        if (lid_x == 0) {
            do {
            } while (atomic_cmpxchg((volatile __global uint*)(pCounter + COUNTER_INDEX), GROUP_ITER,
                                    GROUP_ITER) < 1);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

#if 1

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        }

#else

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp;
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp;
                    }
        }

#endif
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }
    }

#endif
}


#elif KERNEL_METHOD == 2

#ifndef N
#define N   (1)
#endif
#ifndef C
#define C   (1024)
#endif
#ifndef H
#define H   (7)
#endif
#ifndef W
#define W   (7)
#endif
#ifndef K
#define K   (1024)
#endif

#define STRIDE_WEI  C
#define STRIDE_IN   (49)
#define QSTRIDE (C >> 2)
#define ITER    (QSTRIDE >> 5)
#define LDS_WEI_STRIDE  (17)
#define LDS_IN_STRIDE   (132)

#define IN_BATCH_STRIDE     (C * H * W)
#define OUT_BATCH_STRIDE    (K)

void reduce(__local float* buffer, int tid) {
    if ((tid & 63) < 32) {
        buffer[tid + (tid >> 5)] += buffer[tid + 32 + ((tid + 32) >> 5)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((tid & 63) < 16) {
        buffer[tid + (tid >> 5)] += buffer[tid + 16 + ((tid + 16) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 8 + ((tid + 8) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 4 + ((tid + 4) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 2 + ((tid + 2) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 1 + ((tid + 1) >> 5)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void conv1x1_act_pool(
    __global const float* wei,
    __global const float* in,
#ifdef BIAS
    __constant float* bias,
#endif
    __global float* out,
    float slope) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[1024 * 4];
    __local float shared_in[2560 * 4];
    __local float shared_result[512 * 4];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in = (__local float*)shared_in;

    __global const float* pWei = (__global const float*)(wei + ((grid_x & 63) << 4) * STRIDE_WEI +
                                 (lid_x & 127));
    __global const float* pIn = (__global const float*)(in + ((lid_x & 63)));
    __global float* pOut = (__global float*)(out + ((grid_x & 63) << 4) + (lid_x >> 6));
#ifdef BIAS
    __constant float* pBias = (__constant float*)(bias + ((grid_x & 63) << 4) + (lid_x >> 6));
#endif


    float sum[N] = { 0.0f };

    uint offset = ((grid_x & 63) << 7) % STRIDE_WEI;


    for (uchar k = 0; k < ITER; k++, offset = (offset + 128) % STRIDE_WEI) {
        for (uchar i = 0; i < 2; i++) {
            shared_wei[(lid_x & 127) + ((lid_x & 127) >> 5) + ((lid_x >> 7 << 1) + i) * LDS_IN_STRIDE] =
                pWei[((lid_x >> 7 << 1) + i) * STRIDE_WEI + offset];
        }

        for (uchar n = 0; n < N; n++) {
            for (uchar i = 0; i < 8; i++) {
                shared_in[((lid_x >> 6 << 3) + i) * 66 + (lid_x & 63) + ((lid_x & 63) >> 5)] = ((
                            lid_x & 63) < STRIDE_IN ? pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] :
                        0.0f); //pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < 128; j++) {
                sum[n] += pShared_wei[j + (j >> 5) + ((lid_x >> 6) * LDS_IN_STRIDE)] *
                          pShared_in[j * 66 + (lid_x & 63) + ((lid_x & 63) >> 5)];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }


    for (uint n = 0; n < N; n++) {
#ifdef BIAS
        sum[n] += (lid_x & 63) < STRIDE_IN ? pBias[0] : 0.0f;
#endif
        sum[n] *= (sum[n] > 0.0f ? 1.0f : slope);
        shared_result[lid_x + (lid_x >> 5)] = sum[n];
        barrier(CLK_LOCAL_MEM_FENCE);

        reduce(shared_result, lid_x);

        if ((lid_x & 63) == 0) {
            pOut[n * OUT_BATCH_STRIDE] = shared_result[(lid_x) + (lid_x >> 5)] / 49.0f;
        }
    }
}

#elif KERNEL_METHOD == 3

#ifndef N
#define N   (1)
#endif
#ifndef C
#define C   (320)
#endif
#ifndef H
#define H   (7)
#endif
#ifndef W
#define W   (7)
#endif
#ifndef K
#define K   (1280)
#endif

#define PER_ITER_STRIDE (80)
#define STRIDE_WEI  C
#define STRIDE_IN   (H * W)
#define ITER    (STRIDE_WEI / PER_ITER_STRIDE)

#define LDS_WEI_STRIDE  (132)
#define LDS_WEI_COL (128)
#define LDS_WEI_COL_REAL    (80)
#define LDS_WEI_ROW (20)
#define LDS_IN_STRIDE   (66)
#define LDS_IN_COL_REAL (STRIDE_IN)
#define LDS_IN_COL  (64)
#define LDS_IN_ROW  (80)
#define LDS_WEI_READ_ITER   (3)
#define LDS_IN_READ_ITER    (5)
#define PER_WI_TILE_ROW (1)
#define PER_WI_TILE_COL (1)

#define COUNTER_STRIDE  64
#define IN_COL_WG   1
#define GROUP_ITER  (grid_x / COUNTER_STRIDE)
#define WEI_ROW_START   ((grid_x % COUNTER_STRIDE) / IN_COL_WG * LDS_WEI_ROW)
#define WEI_COL_START   (grid_x / COUNTER_STRIDE)
#define WEI_WI_COL_START    (lid_x % LDS_WEI_COL)
#define IN_COL_START    ((grid_x % IN_COL_WG) * LDS_IN_ROW)
#define IN_WI_COL_START (lid_x % COUNTER_STRIDE)
#define OUT_WI_ROW_START    (lid_x / STRIDE_IN)
#define OUT_WI_COL_START    (lid_x & 15)
#define OUT_PER_WG  (STRIDE_IN * LDS_WEI_ROW)
#define OUT_PER_WG_REAL (LDS_WEI_ROW)
#define OUT_WI_INDEX    (lid_x)
#define LDS_WEI_WI  (lid_x % LDS_WEI_COL)
#define LDS_WEI_ROW_START   ((lid_x / LDS_WEI_COL))
#define LDS_IN_ROW_START    ((lid_x >> 6) * LDS_IN_READ_ITER)
#define COMPUTE_WEI_WI_INDEX    (lid_x / STRIDE_IN)
#define COMPUTE_IN_WI_INDEX (lid_x % STRIDE_IN)
#define OFFSET  ((grid_x % COUNTER_STRIDE) * PER_ITER_STRIDE % STRIDE_WEI)

#define IN_BATCH_STRIDE     (C * H * W)
#define OUT_BATCH_STRIDE    (K)

void reduce(__local float* buffer, uint tid, uint start, uint upper) {
    tid += start;

    if (tid < upper) {
        if ((tid & 63) < 32) {
            buffer[tid + (tid >> 5)] += buffer[tid + 32 + ((tid + 32) >> 5)];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < upper) {
        if ((tid & 63) < 16) {
            buffer[tid + (tid >> 5)] += buffer[tid + 16 + ((tid + 16) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 8 + ((tid + 8) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 4 + ((tid + 4) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 2 + ((tid + 2) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 1 + ((tid + 1) >> 5)];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void conv1x1_act_pool(
    __global const float* wei,
    __global const float* in,
#ifdef BIAS
    __constant float* bias,
#endif
    __global float* out,
    float slope) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[1024 * 4];
    __local float shared_in[2560 * 4];
    __local float shared_result[512 * 4];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in = (__local float*)shared_in;

    __global const float* pWei = (__global const float*)(wei + (WEI_ROW_START * STRIDE_WEI +
                                 WEI_WI_COL_START));//
    __global const float* pIn = (__global const float*)(in + (IN_WI_COL_START));//
    __global float* pOut = (__global float*)(out + (WEI_ROW_START + lid_x));//
#ifdef BIAS
    __constant float* pBias = (__constant float*)(bias + (WEI_ROW_START + OUT_WI_ROW_START));//
#endif


    float sum[N] = { 0.0f };

    uint offset = OFFSET;

    for (uint i = 0; i < 2; i++) {
        shared_result[lid_x + i * 1024] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % STRIDE_WEI) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            if (LDS_WEI_ROW_START + i * 8 < LDS_WEI_ROW) {
                shared_wei[WEI_WI_COL_START + (WEI_WI_COL_START >> 5) + (LDS_WEI_ROW_START + i * 8) * LDS_WEI_STRIDE]
                    = (LDS_WEI_WI < LDS_WEI_COL_REAL ? pWei[(LDS_WEI_ROW_START + i * 8) * STRIDE_WEI + offset] : 0.0f);
            }
        }

        for (uchar n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START + (IN_WI_COL_START >> 5)] =
                    (IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] :
                     0.0f); //pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                sum[n] += (lid_x < OUT_PER_WG ?
                           pShared_wei[j + (j >> 5) + (COMPUTE_WEI_WI_INDEX * LDS_WEI_STRIDE)] *
                           pShared_in[j * LDS_IN_STRIDE + COMPUTE_IN_WI_INDEX + (COMPUTE_IN_WI_INDEX >> 5)] : 0.0f);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }


    for (uint n = 0; n < N; n++) {
#ifdef BIAS
        sum[n] += lid_x < OUT_PER_WG ? pBias[0] : 0.0f;
#endif
        sum[n] *= (sum[n] > 0.0f ? 1.0f : slope);
        shared_result[(COMPUTE_WEI_WI_INDEX * 64 + COMPUTE_IN_WI_INDEX) + ((COMPUTE_WEI_WI_INDEX * 64 + COMPUTE_IN_WI_INDEX) >> 5)]
            = sum[n];
        barrier(CLK_LOCAL_MEM_FENCE);

        reduce(shared_result, lid_x, 0, K);
        reduce(shared_result, lid_x, 1024, K);

        if (lid_x < OUT_PER_WG_REAL) {
            pOut[n * OUT_BATCH_STRIDE] = shared_result[(lid_x << 6) + (lid_x << 6 >> 5)] / 49.0f;
        }
    }

}

#elif KERNEL_METHOD == 4
#if GLOBAL_SPLITU == 1
#define ATOMIC  0
#else
#define ATOMIC  2
#endif

///////////////////////////////////////////
#ifndef LOCAL_X
#define LOCAL_X     256
#endif

#ifndef METHOD
#define METHOD  1
#endif

#ifndef BRANCH
#if N == 1
#define BRANCH  1
#elif N == 2
#define BRANCH  3
#endif
#endif

#define GSU_MINUS_ONE   (GLOBAL_SPLITU - 1)
#define BARRIER_MARK    (0xffffffff)
#define STRIDE_WEI  (C)
#define ALIGNED_C   (C + GLOBAL_SPLITU * PER_ITER_STRIDE - 1) / (GLOBAL_SPLITU * PER_ITER_STRIDE) * (GLOBAL_SPLITU * PER_ITER_STRIDE)
#if STRIDE == 2
#define OW  (W / 2)
#define OH  (H / 2)
#define STRIDE_IN_REAL  (H * W)
#else
#define OW  (W)
#define OH  (H)
#endif
#define STRIDE_IN   (OH * OW)


#define WG_PER_IN   ((STRIDE_IN + TILE_COL - 1) / TILE_COL)
#define WG_PER_WEI  ((K + TILE_ROW - 1) / TILE_ROW)

#define QSTRIDE (ALIGNED_C / GLOBAL_SPLITU)
#define ITER    (QSTRIDE / PER_ITER_STRIDE)

#if BRANCH == 1 || BRANCH == 2
#define LDS_WEI_COL (PER_ITER_STRIDE)
#define LDS_WEI_STRIDE  (LDS_WEI_COL + (LDS_WEI_COL + 31) / 32)
#define LDS_WEI_ROW (TILE_ROW)
#define WEI_READ_LINE   LDS_WEI_COL
#elif BRANCH == 3 || BRANCH == 4
#define LDS_WEI_COL (TILE_ROW)
#define LDS_WEI_STRIDE  (LDS_WEI_COL + (LDS_WEI_COL + 31) / 32)
#define LDS_WEI_ROW (PER_ITER_STRIDE)
#define WEI_READ_LINE   LDS_WEI_ROW
#endif
#if BRANCH == 1 || BRANCH == 3
#define LDS_IN_COL  (PER_ITER_STRIDE)
#define LDS_IN_STRIDE   (LDS_IN_COL + (LDS_IN_COL + 31) / 32)
#define LDS_IN_ROW  (TILE_COL)
#define IN_READ_LINE    LDS_IN_ROW
#elif BRANCH == 2 || BRANCH == 4
#define LDS_IN_COL  (TILE_COL)
#define LDS_IN_STRIDE   (LDS_IN_COL + (LDS_IN_COL + 31) / 32)
#define LDS_IN_ROW  (PER_ITER_STRIDE)
#define IN_READ_LINE    LDS_IN_COL
#endif

#define LDS_WEI_READ_ITER   (LDS_WEI_COL * LDS_WEI_ROW / LOCAL_X)
#define LDS_IN_READ_ITER    (LDS_IN_COL * LDS_IN_ROW / LOCAL_X)
#define LDS_WEI_ROW_START   ((lid_x / WEI_READ_LINE) * LDS_WEI_READ_ITER)
#define LDS_IN_ROW_START    ((lid_x / IN_READ_LINE) * LDS_IN_READ_ITER)

#define IN_BATCH_STRIDE     (C * H * W)
#define OUT_BATCH_STRIDE    (K * OH * OW)

#define COUNTER_STRIDE  (WG_PER_IN * WG_PER_WEI)
#define GROUP_ITER  (grid_x / COUNTER_STRIDE)
#define WEI_ROW_START   ((grid_x % COUNTER_STRIDE) / WG_PER_IN * TILE_ROW)
#define WEI_COL_START   (grid_x / COUNTER_STRIDE)
#define WEI_WI_COL_START    (lid_x % PER_ITER_STRIDE)
#if STRIDE == 2
#define IN_COL_START_REAL   (((grid_x % WG_PER_IN) * TILE_COL) + (lid_x % TILE_COL))
#define IN_WI_COL_START_REAL    (IN_COL_START_REAL / OW * OW * 4 + IN_COL_START_REAL % OW * 2)
#define IN_COL_START    ((grid_x % WG_PER_IN) * TILE_COL)
#define IN_WI_COL_START (lid_x % TILE_COL)
#else
#define IN_COL_START    ((grid_x % WG_PER_IN) * TILE_COL)
#define IN_WI_COL_START (lid_x % TILE_COL)
#endif
#define OUT_WI_PER_ROW      (TILE_COL / PER_WI_TILE_COL)
#define OUT_WI_ROW_START    (lid_x / OUT_WI_PER_ROW * PER_WI_TILE_ROW)
#define OUT_WI_COL_START    ((lid_x % OUT_WI_PER_ROW) * PER_WI_TILE_COL)



#define OFFSET_METHOD 0
#if OFFSET_METHOD == 0
#define OFFSET  ((grid_x % COUNTER_STRIDE * STRIDE_WEI / 4096 * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 1
#define OFFSET  (((grid_x / COUNTER_STRIDE) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 2
#define OFFSET  (((grid_x % COUNTER_STRIDE) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 3
#define OFFSET  (((grid_x / WG_PER_WEI) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 4
#define OFFSET  (((grid_x % WG_PER_WEI) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 5
#define OFFSET  (((grid_x / WG_PER_IN) * PER_ITER_STRIDE) % split_stride)
#elif OFFSET_METHOD == 6
#define OFFSET  (((grid_x % WG_PER_IN) * PER_ITER_STRIDE) % split_stride)
#endif


#define COUNTER_INDEX   (grid_x % COUNTER_STRIDE)
#define LOCAL_STRIDE    (LOCAL_X / PER_ITER_STRIDE)

#define NOFILLED

//global: (LOCAL_X * WG_PER_IN * WG_PER_WEI * GLOBAL_SPLITU, 1, 1)

__attribute__((reqd_work_group_size(LOCAL_X, 1, 1)))
__kernel void conv1x1_act(
    __global const float* wei,
    __global const float* in,
#if BIAS == 1
    __constant float* bias,
#endif
    __global float* out,
    float slope, uint C, uint H, uint W, uint K) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[LDS_WEI_ROW * LDS_WEI_STRIDE];
    __local float shared_in[LDS_IN_ROW * LDS_IN_STRIDE];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in = (__local float*)shared_in;

    __global const float* pWei = (__global const float*)(wei + WEI_ROW_START * STRIDE_WEI +
                                 WEI_COL_START * QSTRIDE + WEI_WI_COL_START);//
#if STRIDE == 2
    __global const float* pIn = (__global const float*)(in + (IN_WI_COL_START_REAL) + WEI_COL_START *
                                QSTRIDE * STRIDE_IN_REAL);//
#else
    __global const float* pIn = (__global const float*)(in + (IN_COL_START + IN_WI_COL_START) +
                                WEI_COL_START * QSTRIDE * STRIDE_IN);//
#endif
    __global float* pOut = (__global float*)(out + (IN_COL_START + OUT_WI_COL_START) +
                           (WEI_ROW_START + OUT_WI_ROW_START) * STRIDE_IN);//

#if ATOMIC == 2
    volatile __global uint* pCounter = (volatile __global uint*)(out + OUT_BATCH_STRIDE * N);//
#endif

#if BIAS == 1
    __constant float* pBias = (__constant float*)(bias + WEI_ROW_START + OUT_WI_ROW_START);//
#endif

    uint split_stride;

    if ((GROUP_ITER + 1) * QSTRIDE < C) {
        split_stride = QSTRIDE;
    } else if ((GROUP_ITER + 1) * QSTRIDE - C < QSTRIDE) {
        split_stride = C - (GROUP_ITER) * QSTRIDE;
    } else {
        split_stride = 0;
    }

    uint iter = (split_stride + PER_ITER_STRIDE - 1) / PER_ITER_STRIDE;

    uint remainder = C % PER_ITER_STRIDE;
    remainder = (remainder == 0 ? PER_ITER_STRIDE : remainder);
    uint per_iter_stride;

    float previous_value;
    uint prevVal;
    uint newVal;

#if ATOMIC == 2

    if (grid_x < COUNTER_STRIDE) {
        if (lid_x == 0) {
            *(pCounter + COUNTER_INDEX) = 0;
        }
    }

#endif

    float sum[N][PER_WI_TILE_ROW][PER_WI_TILE_COL] = { { { 0.0f } } };

    uint offset = 0;//OFFSET;

#if BRANCH == 1

    for (uint k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        per_iter_stride = ((k == iter - 1 && GROUP_ITER == GSU_MINUS_ONE) ? remainder : PER_ITER_STRIDE);

        for (uint i = 0; i < LDS_WEI_READ_ITER; i++) {
#if 1
            shared_wei[WEI_WI_COL_START + (((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * LDS_WEI_STRIDE] =
                pWei[(((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * STRIDE_WEI + offset];
            prefetch(pWei + (((lid_x / WEI_READ_LINE) + i * LOCAL_STRIDE)) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
#else
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            prefetch(pWei + (LDS_WEI_ROW_START + i) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
#endif
        }

        for (uint n = 0; n < N; n++) {
            for (uint i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN_REAL -
                         IN_WI_COL_START_REAL + IN_COL_START_REAL / OW * OW * 4 + n * IN_BATCH_STRIDE, 64);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN -
                         IN_WI_COL_START + n * IN_BATCH_STRIDE, 32);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint j = 0; j < per_iter_stride; j++) {
                for (uint m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uint l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] += pShared_wei[j + (OUT_WI_ROW_START + m) * LDS_WEI_STRIDE] *
                                        pShared_in[j + (OUT_WI_COL_START + l) * LDS_IN_STRIDE];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 2

    for (uint k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uint i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uint n = 0; n < N; n++) {
            for (uint i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint j = 0; j < PER_ITER_STRIDE; j++) {
                for (uint m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uint l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] += pShared_wei[j + (OUT_WI_ROW_START + m) * LDS_WEI_STRIDE] *
                                        pShared_in[j * LDS_IN_STRIDE + (OUT_WI_COL_START + l)];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 3

    for (uint k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uint i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * LDS_WEI_STRIDE + (LDS_WEI_ROW_START + i) + (LDS_WEI_ROW_START + i) / 32]
                = pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            prefetch(pWei + (LDS_WEI_ROW_START + i) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % split_stride, 32);
        }

        for (uint n = 0; n < N; n++) {
            for (uint i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN_REAL -
                         IN_WI_COL_START_REAL + IN_COL_START_REAL / OW * OW * 4 + n * IN_BATCH_STRIDE, 64);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % split_stride) * STRIDE_IN -
                         IN_WI_COL_START + n * IN_BATCH_STRIDE, 32);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint j = 0; j < PER_ITER_STRIDE; j++) {
                for (uint l = 0; l < PER_WI_TILE_COL; l++)
                    for (uint m = 0; m < PER_WI_TILE_ROW; m++) {
                        sum[n][m][l] +=
                            pShared_wei[j * LDS_WEI_STRIDE + (OUT_WI_ROW_START + m) + (OUT_WI_ROW_START + m) / 32] *
                            pShared_in[j + (OUT_WI_COL_START + l) * LDS_IN_STRIDE];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 4

    for (uint k = 0; k < iter; k++, offset = (offset + PER_ITER_STRIDE) % split_stride) {
        for (uint i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * LDS_WEI_STRIDE + (LDS_WEI_ROW_START + i) + (LDS_WEI_ROW_START + i) / 32]
                = pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uint n = 0; n < N; n++) {
            for (uint i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint j = 0; j < PER_ITER_STRIDE; j++) {
                for (uint m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uint l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] +=
                            pShared_wei[j * LDS_WEI_STRIDE + (OUT_WI_ROW_START + m) + (OUT_WI_ROW_START + m) / 32] *
                            pShared_in[j * LDS_IN_STRIDE + (OUT_WI_COL_START + l)];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#endif

    uint out_width = STRIDE_IN;
    uint out_height = K;
    uint mod_width = out_width % PER_WI_TILE_COL;
    uint mod_height = out_height % PER_WI_TILE_ROW;
    uint per_wi_tile_col = 0;
    uint per_wi_tile_row = 0;

    if (mod_width == 0) {
        if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            per_wi_tile_col = PER_WI_TILE_COL;
        }
    } else {
        if (IN_COL_START + OUT_WI_COL_START + mod_width < STRIDE_IN) {
            per_wi_tile_col = PER_WI_TILE_COL;
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            per_wi_tile_col = mod_width;
        }
    }

    if (mod_height == 0) {
        if (WEI_ROW_START + OUT_WI_ROW_START < out_height) {
            per_wi_tile_row = PER_WI_TILE_ROW;
        }
    } else {
        if (WEI_ROW_START + OUT_WI_ROW_START + mod_height < out_height) {
            per_wi_tile_row = PER_WI_TILE_ROW;
        } else if (WEI_ROW_START + OUT_WI_ROW_START < out_height) {
            per_wi_tile_row = mod_height;
        }
    }

#if ATOMIC == 0

    for (uint n = 0; n < N; n++)
        for (uint j = 0; j < per_wi_tile_row; j++)
            for (uint i = 0; i < per_wi_tile_col; i++) {
#if BIAS == 1
                float temp = sum[n][j][i] + pBias[j];
#else
                float temp = sum[n][j][i];
#endif
                temp *= (temp > 0.0f ? 1.0f : slope);
                pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
            }

#elif ATOMIC == 2

    if (GROUP_ITER == 0) {
        for (uint n = 0; n < N; n++)
            for (uint j = 0; j < per_wi_tile_row; j++)
                for (uint i = 0; i < per_wi_tile_col; i++) {
#if BIAS == 1
                    *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
#else
                    *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
#endif
                }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }
    } else if (GROUP_ITER == GSU_MINUS_ONE) {
#ifdef NOFILLED

        if (lid_x == 0) {
            do {
            } while (atomic_cmpxchg((volatile __global uint*)(pCounter + COUNTER_INDEX), GROUP_ITER,
                                    GROUP_ITER) < GROUP_ITER);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        for (uint n = 0; n < N; n++)
            for (uint j = 0; j < per_wi_tile_row; j++)
                for (uint i = 0; i < per_wi_tile_col; i++) {
                    float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                    temp += sum[n][j][i];
                    *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp * (temp > 0.0f ? 1.0f : slope);
                }
    } else {
        if (lid_x == 0) {
            do {
            } while (atomic_cmpxchg((volatile __global uint*)(pCounter + COUNTER_INDEX), GROUP_ITER,
                                    GROUP_ITER) < 1);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint n = 0; n < N; n++)
            for (uint j = 0; j < per_wi_tile_row; j++)
                for (uint i = 0; i < per_wi_tile_col; i++) {
                    do {
                        previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        prevVal = as_uint(previous_value);
                        newVal = as_uint(sum[n][j][i] + previous_value);
                    } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                            newVal) != prevVal);
                }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }
    }

#endif
}

#elif KERNEL_METHOD == 5
#elif KERNEL_METHOD == 6
#elif KERNEL_METHOD == 7
#elif KERNEL_METHOD == 8
#elif KERNEL_METHOD == 9
#endif

#endif // #ifdef MACRO
