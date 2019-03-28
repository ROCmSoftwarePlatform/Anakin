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
#ifndef N
#define N   1
#endif

#ifndef WIDTH
#define WIDTH (1024)
#endif

#ifndef OUTPUT
#define OUTPUT 21841
#endif

#ifndef KERNEL_METHOD
#define KERNEL_METHOD 0
#endif

#if KERNEL_METHOD == 1
#define ITER (WIDTH >> 6)

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b,
#ifdef BIAS
    __global const float* bias,
#endif
    __global float* c) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float result[N][66];

    __global const float* pB = (__global const float*)(b + grid_x * WIDTH + lid_x);
    __global float* pC;


    uint offset = (grid_x >> 2 << 6) % WIDTH;

    float sum[N] = { 0.0f };

    if (grid_x < OUTPUT) {
        for (uint i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
            for (uint n = 0; n < N; n++) {
                result[n][lid_x + (lid_x >> 5)] = a[offset + lid_x + n * WIDTH];
            }

            pB = (__global const float*)(b + grid_x * WIDTH + lid_x);
            pC = (__global float*)(c + grid_x);

            for (uint n = 0; n < N; n++) {
                sum[n] += result[n][lid_x + (lid_x >> 5)] * pB[offset];
            }

        }

        for (uint n = 0; n < N; n++) {
            result[n][lid_x] = sum[n];

            if (lid_x < 32) {
                result[n][lid_x] += result[n][lid_x + 32];
            }

            if (lid_x < 16) {
                result[n][lid_x] += result[n][lid_x + 16];
            }

            if (lid_x < 8) {
                result[n][lid_x] += result[n][lid_x + 8];
            }

            if (lid_x < 4) {
                result[n][lid_x] += result[n][lid_x + 4];
            }

            if (lid_x < 2) {
                result[n][lid_x] += result[n][lid_x + 2];
            }

            if (lid_x < 1) {
                result[n][lid_x] += result[n][lid_x + 1];
            }

            if (lid_x == 0) {
#ifdef BIAS
                pC[n * OUTPUT] = bias[grid_x] + result[n][0];
#else
                pC[n * OUTPUT] = result[n][0];
#endif
            }
        }
    }
}
#elif KERNEL_METHOD == 2
#define ITER (WIDTH >> 6)

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __constant float* a, __global const float* b,
#ifdef BIAS
    __global const float* bias,
#endif
    __global float* c
#ifndef NO_SLOPE
    ,
    float slope
#endif
) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float result[N][66];

    if (grid_x < OUTPUT) {
        __constant float* pA;
        __global const float* pB = (__global const float*)(b + grid_x * WIDTH + lid_x);
        __global float* pC;

        pA = (__constant float*)(a + lid_x);
        pC = (__global float*)(c + grid_x);

        uint offset = (grid_x >> 2 << 6) % WIDTH;

        float sum[N] = { 0.0f };

        for (uint i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
            for (uint n = 0; n < N; n++) {
                sum[n] += pA[offset + n * WIDTH] * pB[offset];
            }
        }

        for (uint n = 0; n < N; n++) {
            result[n][lid_x] = sum[n];

            if (lid_x < 32) {
                result[n][lid_x] += result[n][lid_x + 32];
            }

            if (lid_x < 16) {
                result[n][lid_x] += result[n][lid_x + 16];
            }

            if (lid_x < 8) {
                result[n][lid_x] += result[n][lid_x + 8];
            }

            if (lid_x < 4) {
                result[n][lid_x] += result[n][lid_x + 4];
            }

            if (lid_x < 2) {
                result[n][lid_x] += result[n][lid_x + 2];
            }

            if (lid_x < 1) {
                result[n][lid_x] += result[n][lid_x + 1];
            }

            if (lid_x == 0) {
#ifdef BIAS
                pC[n * OUTPUT] = bias[grid_x] + result[n][0];
#else
                pC[n * OUTPUT] = result[n][0];
#endif
#ifndef NO_SLOPE
                pC[n * OUTPUT] *= (pC[n * OUTPUT] > 0 ? 1.0f : slope);
#endif
            }
        }
    }
}
#elif KERNEL_METHOD == 3
#define HSTRIDE (WIDTH >> 1)
#define ITER (WIDTH >> 7)

void reduce(__local float* buffer, int tid) {
    if (tid < 64) {
        buffer[tid] += buffer[tid + 64];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32) {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 16) {
        buffer[tid << 2] += buffer[(tid << 2) + 2];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 8) {
        buffer[tid << 3] += buffer[(tid << 3) + 4];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 4) {
        buffer[tid << 4] += buffer[(tid << 4) + 8];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c) {
    __local float shared_a[129];
    __local float shared_b[8][65];

    __local float result[2][129];

    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __global const float* pA = (__global const float*)(a + (grid_x >> 9) * WIDTH);
    __global const float* pB = (__global const float*)(b + ((grid_x & 511) << 3) * WIDTH);

    int offset = ((grid_x << 6) + ((lid_x >> 6) * HSTRIDE) + (lid_x & 63)) % WIDTH;

    int temp_offset = offset;

    for (int l = 0; l < 2; l++, offset = temp_offset) {
        float sum = 0.0f;

        for (int i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
            shared_a[lid_x] = pA[offset];

            for (int j = l * 4; j < (l + 1) * 4; j++) {
                shared_b[(lid_x >> 6 << 2) + (j & 3)][(lid_x & 63)] = pB[offset + j * WIDTH];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int k = 0; k < 4; k++) {
                sum += shared_a[(lid_x >> 6 << 6) + ((lid_x & 15) << 2) + k] *
                       shared_b[(lid_x >> 6 << 2) + ((lid_x & 63) >> 4)][((lid_x & 15) << 2) + k];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        result[l][lid_x] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        reduce(result[l], lid_x);
    }

    if (lid_x < 8) {
        c[(grid_x << 3) + lid_x] =
            result[lid_x >> 2][(lid_x & 3) << 4] + bias[(grid_x << 3) + lid_x];
    }
}
#elif KERNEL_METHOD == 4
#define ITER (WIDTH >> 6)

void reduce(__local float* buffer, int tid) {
    if (tid < 32) {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }

    if (tid < 16) {
        buffer[tid << 2] += buffer[(tid << 2) + 2];
    }
}

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c) {
    __local float shared_a[2][66];
    __local float shared_b[8][66];
    __local float result[65];

    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __global const float* pA = (__global const float*)(a);
    __global const float* pB = (__global const float*)(b + (grid_x << 3) * WIDTH);

    int offset = ((grid_x << 6)) % WIDTH;
    float sum  = 0.0f;

    for (int i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
        for (int j = 0; j < 2; j++) {
            shared_a[j][lid_x + (lid_x >> 5)] = pA[offset + lid_x + j * WIDTH];
        }

        for (int j = 0; j < 8; j++) {
            shared_b[j][lid_x + (lid_x >> 5)] = pB[offset + lid_x + j * WIDTH];
        }

        for (int k = 0; k < 16; k++) {
            sum += shared_a[lid_x >> 5][((lid_x & 3) << 4) + k + ((((lid_x & 3) << 4) + k) >> 5)] *
                   shared_b[(lid_x & 31) >> 2]
                   [((lid_x & 3) << 4) + k + ((((lid_x & 3) << 4) + k) >> 5)];
        }
    }

    result[lid_x] = sum;
    reduce(result, lid_x);

    if (lid_x < 2) {
        float8 out;
        float* pOut = (float*)&out;

        for (int i = 0; i < 8; i++) {
            pOut[i] = result[((lid_x * 8 + i) << 2)] + bias[(grid_x << 3) + i];
        }

        __global float8* pC = (__global float8*)(c + (grid_x << 3) + lid_x * OUTPUT);
        *pC                 = out;
    }
}
#elif KERNEL_METHOD == 5
#define ITER (WIDTH >> 6)

void reduce(__local float* buffer, int tid) {
    if (tid < 32) {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }

    if (tid < 16) {
        buffer[tid << 2] += buffer[(tid << 2) + 2];
    }
}

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c) {
    __local float shared_a[4][66];
    __local float shared_b[4][66];
    __local float result[65];

    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __global const float* pA = (__global const float*)(a);
    __global const float* pB = (__global const float*)(b + (grid_x << 2) * WIDTH);

    int offset = ((grid_x << 6)) % WIDTH;
    float sum  = 0.0f;

    for (int i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
        for (int j = 0; j < 4; j++) {
            shared_a[j][lid_x + (lid_x >> 5)] = pA[offset + lid_x + j * WIDTH];
            shared_b[j][lid_x + (lid_x >> 5)] = pB[offset + lid_x + j * WIDTH];
        }

        for (int k = 0; k < 16; k++) {
            sum += shared_a[lid_x >> 4][((lid_x & 3) << 4) + k + ((((lid_x & 3) << 4) + k) >> 5)] *
                   shared_b[(lid_x & 15) >> 2]
                   [((lid_x & 3) << 4) + k + ((((lid_x & 3) << 4) + k) >> 5)];
        }
    }

    result[lid_x] = sum;
    reduce(result, lid_x);

    if (lid_x < 4) {
        float4 out;
        float* pOut = (float*)&out;

        for (int i = 0; i < 4; i++) {
            pOut[i] = result[((lid_x * 4 + i) << 2)] + bias[(grid_x << 2) + i];
        }

        __global float4* pC = (__global float4*)(c + (grid_x << 2) + lid_x * OUTPUT);
        *pC                 = out;
    }
}
#elif KERNEL_METHOD == 6
#define ITER (WIDTH >> 6)

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c) {
    __local float shared_a[8][66];
    __local float shared_b[8][66];
    __local float result[65];

    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __global const float* pA = (__global const float*)(a);
    __global const float* pB = (__global const float*)(b + (grid_x << 3) * WIDTH);

    int offset = ((grid_x << 6)) % WIDTH;
    float sum  = 0.0f;

    for (int i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
        for (int j = 0; j < 8; j++) {
            shared_a[j][lid_x + (lid_x >> 5)] = pA[offset + lid_x + j * WIDTH];
        }

        for (int j = 0; j < 8; j++) {
            shared_b[j][lid_x + (lid_x >> 5)] = pB[offset + lid_x + j * WIDTH];
        }

        for (int k = 0; k < 64; k++) {
            sum += shared_a[lid_x >> 3][k + (k >> 5)] * shared_b[(lid_x & 7)][k + (k >> 5)];
        }
    }

    result[lid_x] = sum;

    if (lid_x < 8) {
        float8 out;
        float* pOut = (float*)&out;

        for (int i = 0; i < 8; i++) {
            pOut[i] = result[((lid_x * 8 + i))] + bias[(grid_x << 3) + i];
        }

        __global float8* pC = (__global float8*)(c + (grid_x << 3) + lid_x * OUTPUT);
        *pC                 = out;
    }
}
#elif KERNEL_METHOD == 7
#define QSTRIDE (WIDTH >> 2)
#define ITER (WIDTH >> 7)

__attribute__((reqd_work_group_size(256, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c) {
    __local float shared_a[1536];
    __local float shared_b[2560];
    __local float2* pShared_a = (__local float2*)shared_a;
    __local float4* pShared_b = (__local float4*)shared_b;

    float2 sha;
    float* pSha = (float*)&sha;
    float4 shb;
    float* pShb = (float*)&shb;
    float4 sum[2];
    float* pSum = (float*)sum;

    uint lid_x  = get_local_id(0);
    uint grid_x = get_group_id(0);

    __global const float* pA =
        (__global const float*)(a + (lid_x >> 6 << 3) * WIDTH + (grid_x >> 6) * QSTRIDE);
    __global const float* pB =
        (__global const float*)(b + (((grid_x & 63) << 6) + (lid_x >> 6 << 4)) * WIDTH +
                                (grid_x >> 6) * QSTRIDE);
    __global float4* pC    = (__global float4*)(c + ((lid_x >> 4 << 1) * OUTPUT +
                             ((grid_x & 63) << 6) + ((lid_x & 15) << 2)));
    __global float4* pBias = (__global float4*)(bias + ((grid_x & 63) << 6) + ((lid_x & 15) << 2));

    int offset = (((grid_x & 63) << 5)) % QSTRIDE;

    for (uint i = 0; i < 2; i++) {
        sum[i] = 0.0f;
    }

    for (ushort i = 0; i < ITER; i++, offset = (offset + 32) % QSTRIDE) {
        for (uint j = 0; j < 4; j++) {
            shared_a[((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 3)) + ((lid_x & 31)) * 32 +
                     ((((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 3)) +
                       ((lid_x & 31)) * 32) >>
                      5)] = pA[((j << 1) + ((lid_x & 63) >> 5)) * WIDTH + (lid_x & 31) + offset];
        }

        for (uint j = 0; j < 8; j++) {
            shared_b[((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 4)) + ((lid_x & 31)) * 64 +
                     ((((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 4)) +
                       ((lid_x & 31)) * 64) >>
                      5)] = pB[((j << 1) + ((lid_x & 63) >> 5)) * WIDTH + (lid_x & 31) + offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint k = 0; k < 32; k++) {
            for (uint m = 0; m < 2; m++) {
                pSha[m] = shared_a[((lid_x >> 4) << 1) + m + k * 32 +
                                   ((((lid_x >> 4) << 1) + m + k * 32) >> 5)];
            }

            for (uint l = 0; l < 4; l++) {
                pShb[l] = shared_b[((lid_x & 15) << 2) + l + k * 64 +
                                   ((((lid_x & 15) << 2) + l + k * 64) >> 5)];
            }

            for (uint m = 0; m < 2; m++) {
                for (uint l = 0; l < 4; l++) {
                    pSum[m * 4 + l] += pSha[m] * pShb[l];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if ((grid_x >> 6) == 0) {
        for (uint i = 0; i < 2; i++) {
            pC[i * OUTPUT >> 2] = sum[i] + pBias[0];
        }
    } else {
        for (uint i = 0; i < 2; i++) {
            pC[i * OUTPUT >> 2] += sum[i];
        }
    }
}
#elif KERNEL_METHOD == 8
#define ITER (WIDTH >> 6)
#define HWG (384 >> 1)

void reduce(__local float* buffer, int tid) {
    if (tid < 32) {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }
}

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c) {
    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __local float shared_a[4][65];
    __local float shared_b[8][65];
    __local float result[65];

    __global const float* pA =
        (__global const float*)(a + ((grid_x / HWG) * (STRIDE << 2)));
    __global const float* pB = (__global const float*)(b);

    int offset = (((grid_x % HWG) << 6)) % STRIDE;

    float sum = 0.0f;

    for (int i = 0; i < ITER; i++, offset = (offset + 64) % STRIDE) {
        for (int j = 0; j < 4; j++) {
            shared_a[j][lid_x] = pA[offset + j * STRIDE + lid_x];
        }

        for (int j = 0; j < 8; j++) {
            shared_b[j][(lid_x)] =
                ((j + ((grid_x % HWG) << 3)) * STRIDE + (offset + lid_x) < OUTPUT * STRIDE
                 ? pB[(j + ((grid_x % HWG) << 3)) * STRIDE + (offset + lid_x)]
                 : 0.0f);
        }

        for (int k = 0; k < 32; k++) {
            sum += shared_a[lid_x >> 4][((lid_x & 1) << 5) + k] *
                   shared_b[((lid_x & 15) >> 1)][((lid_x & 1) << 5) + k];
        }
    }

    result[lid_x] = sum;
    reduce(result, lid_x);

    if (lid_x < 32 && ((grid_x % HWG) << 3) + (lid_x & 7) < OUTPUT) {
        int out_offset =
            ((grid_x / HWG << 2) + (lid_x >> 3)) * OUTPUT + ((grid_x % HWG) << 3) + (lid_x & 7);
        c[out_offset] = bias[((grid_x % HWG) << 3) + (lid_x & 7)] + result[(lid_x << 1)];
    }
}
#elif KERNEL_METHOD == 9
#define ITER (WIDTH >> 6)

void reduce(__local float* buffer, int tid) {
    if (tid < 32) {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }
}

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c) {
    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __local float shared_a[4][65];
    __local float shared_b[8][65];
    __local float result[65];

    __global const float* pA = (__global const float*)(a + ((grid_x >> 7) * (WIDTH << 2)));
    __global const float* pB = (__global const float*)(b);

    int offset = (((grid_x & 127) << 6)) % STRIDE;

    float sum = 0.0f;

    for (int i = 0; i < ITER; i++, offset = (offset + 64) % STRIDE) {
        for (int j = 0; j < 4; j++) {
            shared_a[j][lid_x] = pA[offset + j * STRIDE + lid_x];
        }

        for (int j = 0; j < 8; j++) {
            shared_b[j][(lid_x)] =
                ((j + ((grid_x & 127) << 3)) * STRIDE + (offset + lid_x) < OUTPUT * STRIDE
                 ? pB[(j + ((grid_x & 127) << 3)) * STRIDE + (offset + lid_x)]
                 : 0.0f);
        }

        for (int k = 0; k < 32; k++) {
            sum += shared_a[lid_x >> 4][((lid_x & 1) << 5) + k] *
                   shared_b[((lid_x & 15) >> 1)][((lid_x & 1) << 5) + k];
        }
    }

    result[lid_x] = sum;
    reduce(result, lid_x);

    if (lid_x < 32 && ((grid_x & 127) << 3) + (lid_x & 7) < OUTPUT) {
        int out_offset =
            ((grid_x >> 7 << 2) + (lid_x >> 3)) * OUTPUT + ((grid_x & 127) << 3) + (lid_x & 7);
        c[out_offset] = bias[((grid_x & 127) << 3) + (lid_x & 7)] + result[(lid_x << 1)];
    }
}
#elif KERNEL_METHOD == 10
#ifndef ATOMIC
#define ATOMIC  32
#endif

//#define ROW_ALIGN (64 / ATOMIC)

#ifndef LOCAL_SIZE
#define LOCAL_SIZE  64
#endif

#define LOCAL_MEMORY    (LOCAL_SIZE + (LOCAL_SIZE >> 5))

#ifndef N
#define N   1
#endif

#ifndef WIDTH
#define WIDTH (1024)
#endif

#define WORKLOAD    (LOCAL_SIZE * ATOMIC)
#define ITER ((WIDTH + WORKLOAD - 1) / WORKLOAD)


#ifndef OUTPUT
#define OUTPUT 21841
#endif



__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b,
#ifdef BIAS
    __global const float* bias,
#endif
    __global float* c,
    __global float* gAtomicLock) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);
    uint col_id = grid_x / ATOMIC;
    uint atomic_id = grid_x % ATOMIC;

    __local float result[2][LOCAL_MEMORY];

    __global const float* pB;
    __global float* pC;

    volatile __global uint* pCounter = (volatile __global uint*)(gAtomicLock);

    uint wave_stride = ITER * WORKLOAD / (WORKLOAD >> 6);
    uint offset = (grid_x >> 2 << 6) % wave_stride;
    uint wave_id = (atomic_id << 4) + (lid_x >> 6);

    float sum = 0.0f;

    float previous_value;
    uint prevVal;
    uint newVal;

    if (grid_x % ATOMIC == 0) {
        for (uint n = 0; n < N; n++)
            if (lid_x == 0) {
                *(pCounter + col_id + n * OUTPUT) = 0;
            }
    }

    if (col_id < OUTPUT) {
        for (uint n = 0; n < N; n++) {

            pC = (__global float*)(c + col_id);

            sum = 0.0f;

            for (uint i = 0; i < ITER; i++, offset = (offset + 64) % wave_stride) {
                //result[n % 2][lid_x + (lid_x >> 5)] = (offset + wave_id * wave_stride + (lid_x & 63) < WIDTH ? a[offset + wave_id * wave_stride + (lid_x & 63) + n * WIDTH] : 0.0f);
                result[n % 2][lid_x + (lid_x >> 5)] = a[offset + wave_id * wave_stride + (lid_x & 63) + n * WIDTH];

                pB = (__global const float*)(b + col_id * WIDTH + offset + wave_id * wave_stride + (lid_x & 63));

                sum += (offset + wave_id * wave_stride + (lid_x & 63) < WIDTH ?
                        result[n % 2][lid_x + (lid_x >> 5)] * pB[0] : 0.0f);
            }

            result[n % 2][lid_x + (lid_x >> 5)] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint i = LOCAL_SIZE >> 1; i >= 64; i >>= 1) {
                if (lid_x < i) {
                    result[n % 2][lid_x + (lid_x >> 5)] += result[n % 2][lid_x + i + ((lid_x + i) >> 5)];
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid_x < 32) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 33];
            }

            if (lid_x < 16) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 16];
            }

            if (lid_x < 8) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 8];
            }

            if (lid_x < 4) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 4];
            }

            if (lid_x < 2) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 2];
            }

            if (lid_x < 1) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 1];
            }

            if (grid_x % ATOMIC == 0) {
#ifdef BIAS
                pC[n * OUTPUT] = bias[col_id] + result[n % 2][0];
#else
                pC[n * OUTPUT] = result[n % 2][0];
#endif

                barrier(CLK_GLOBAL_MEM_FENCE);

                if (lid_x == 0) {
                    atomic_inc(pCounter + col_id + n * OUTPUT);
                }
            } else {
                if (lid_x == 0) {
                    do {
                    } while (atomic_cmpxchg((volatile __global uint*)(pCounter + col_id + n * OUTPUT), 1, 1) == 0);
                }

                barrier(CLK_LOCAL_MEM_FENCE);


                if (lid_x == 0) {
                    do {
                        previous_value = pC[n * OUTPUT];
                        prevVal = as_uint(previous_value);
                        newVal = as_uint(result[n % 2][0] + previous_value);
                    } while (atomic_cmpxchg((__global uint*)(pC + n * OUTPUT), prevVal, newVal) != prevVal);
                }
            }
        }
    }
}
#elif KERNEL_METHOD == 11
#ifndef LOCAL_SIZE
#define LOCAL_SIZE  64
#endif

#define LOCAL_MEMORY    (LOCAL_SIZE + (LOCAL_SIZE >> 5))

#ifndef N
#define N   1
#endif

#ifndef WIDTH
#define WIDTH (1024)
#endif


#define ITER ((WIDTH + LOCAL_SIZE - 1) / LOCAL_SIZE)

#ifndef OUTPUT
#define OUTPUT 21841
#endif

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b,
#ifdef BIAS
    __global const float* bias,
#endif
    __global float* c) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float result[2][LOCAL_MEMORY];

    __global const float* pB; // correct
    __global float* pC;

    uint wave_stride = ITER * LOCAL_SIZE / (LOCAL_SIZE >> 6);
    uint offset = (grid_x >> 2 << 6) % wave_stride;
    uint wave_id = (lid_x >> 6);

    float sum = 0.0f;

    if (grid_x < OUTPUT) {
        for (uint n = 0; n < N; n++) {
            sum = 0.0f;

            for (uint i = 0; i < ITER; i++, offset = (offset + 64) % wave_stride) {
                //result[n % 2][lid_x + (lid_x >> 5)] = (offset + wave_id * wave_stride + (lid_x & 63) < WIDTH ? a[offset + wave_id * wave_stride + (lid_x & 63) + n * WIDTH] : 0.0f);
                result[n % 2][lid_x + (lid_x >> 5)] = a[offset + wave_id * wave_stride + (lid_x & 63) + n * WIDTH];

                pB = (__global const float*)(b + grid_x * WIDTH + offset + wave_id * wave_stride + (lid_x & 63));
                pC = (__global float*)(c + grid_x);

                sum += (offset + wave_id * wave_stride + (lid_x & 63) < WIDTH ?
                        result[n % 2][lid_x + (lid_x >> 5)] * pB[0] : 0.0f);
            }

            result[n % 2][lid_x + (lid_x >> 5)] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint i = LOCAL_SIZE >> 1; i >= 64; i >>= 1) {
                if (lid_x < i) {
                    result[n % 2][lid_x + (lid_x >> 5)] += result[n % 2][lid_x + i + ((lid_x + i) >> 5)];
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }


            if (lid_x < 32) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 33];
            }

            if (lid_x < 16) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 16];
            }

            if (lid_x < 8) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 8];
            }

            if (lid_x < 4) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 4];
            }

            if (lid_x < 2) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 2];
            }

            if (lid_x < 1) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 1];
            }

            if (lid_x == 0) {
#ifdef BIAS
                pC[n * OUTPUT] = bias[grid_x] + result[n % 2][0];
#else
                pC[n * OUTPUT] = result[n % 2][0];
#endif
            }
        }
    }
}
#endif

#else //#ifndef MACRO

#ifndef KERNEL_METHOD
#define KERNEL_METHOD 1
#endif

#if KERNEL_METHOD == 1
#define ITER (WIDTH >> 6)

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b,
#ifdef BIAS
    __global const float* bias,
#endif
    __global float* c, uint N, uint WIDTH, uint OUTPUT) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float result[2][66];

    __global const float* pB = (__global const float*)(b + grid_x * WIDTH + lid_x);
    __global float* pC;


    uint offset = (grid_x >> 2 << 6) % WIDTH;

    float sum = 0.0f;

    if (grid_x < OUTPUT) {
        for (uint n = 0; n < N; n++) {
            sum = 0.0f;

            for (uint i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
                result[n % 2][lid_x + (lid_x >> 5)] = a[offset + lid_x + n * WIDTH];

                pB = (__global const float*)(b + grid_x * WIDTH + lid_x);
                pC = (__global float*)(c + grid_x);

                sum += result[n % 2][lid_x + (lid_x >> 5)] * pB[offset];
            }

            result[n % 2][lid_x] = sum;

            if (lid_x < 32) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 32];
            }

            if (lid_x < 16) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 16];
            }

            if (lid_x < 8) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 8];
            }

            if (lid_x < 4) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 4];
            }

            if (lid_x < 2) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 2];
            }

            if (lid_x < 1) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 1];
            }

            if (lid_x == 0) {
#ifdef BIAS
                pC[n * OUTPUT] = bias[grid_x] + result[n % 2][0];
#else
                pC[n * OUTPUT] = result[n % 2][0];
#endif
            }
        }
    }
}
#elif KERNEL_METHOD == 2
#define ITER (WIDTH >> 6)

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __constant float* a, __global const float* b,
#ifdef BIAS
    __global const float* bias,
#endif
    __global float* c, uint N, uint WIDTH, uint OUTPUT
#ifndef NO_SLOPE
    ,
    float slope
#endif
) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float result[8][66];

    if (grid_x < OUTPUT) {
        __constant float* pA;
        __global const float* pB = (__global const float*)(b + grid_x * WIDTH + lid_x);
        __global float* pC;

        pA = (__constant float*)(a + lid_x);
        pC = (__global float*)(c + grid_x);

        uint offset = (grid_x >> 2 << 6) % WIDTH;

        float sum[8] = { 0.0f };

        for (uint i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
            for (uint n = 0; n < N; n++) {
                sum[n] += pA[offset + n * WIDTH] * pB[offset];
            }
        }

        for (uint n = 0; n < N; n++) {
            result[n][lid_x] = sum[n];

            if (lid_x < 32) {
                result[n][lid_x] += result[n][lid_x + 32];
            }

            if (lid_x < 16) {
                result[n][lid_x] += result[n][lid_x + 16];
            }

            if (lid_x < 8) {
                result[n][lid_x] += result[n][lid_x + 8];
            }

            if (lid_x < 4) {
                result[n][lid_x] += result[n][lid_x + 4];
            }

            if (lid_x < 2) {
                result[n][lid_x] += result[n][lid_x + 2];
            }

            if (lid_x < 1) {
                result[n][lid_x] += result[n][lid_x + 1];
            }

            if (lid_x == 0) {
#ifdef BIAS
                pC[n * OUTPUT] = bias[grid_x] + result[n][0];
#else
                pC[n * OUTPUT] = result[n][0];
#endif
#ifndef NO_SLOPE
                pC[n * OUTPUT] *= (pC[n * OUTPUT] > 0 ? 1.0f : slope);
#endif
            }
        }

    }
}
#elif KERNEL_METHOD == 3
#define HSTRIDE (WIDTH >> 1)
#define ITER (WIDTH >> 7)

void reduce(__local float* buffer, int tid) {
    if (tid < 64) {
        buffer[tid] += buffer[tid + 64];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32) {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 16) {
        buffer[tid << 2] += buffer[(tid << 2) + 2];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 8) {
        buffer[tid << 3] += buffer[(tid << 3) + 4];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 4) {
        buffer[tid << 4] += buffer[(tid << 4) + 8];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c,
    uint N, uint WIDTH, uint OUTPUT) {
    __local float shared_a[129];
    __local float shared_b[8][65];

    __local float result[2][129];

    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __global const float* pA = (__global const float*)(a + (grid_x >> 9) * WIDTH);
    __global const float* pB = (__global const float*)(b + ((grid_x & 511) << 3) * WIDTH);

    int offset = ((grid_x << 6) + ((lid_x >> 6) * HSTRIDE) + (lid_x & 63)) % WIDTH;

    int temp_offset = offset;

    for (int l = 0; l < 2; l++, offset = temp_offset) {
        float sum = 0.0f;

        for (int i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
            shared_a[lid_x] = pA[offset];

            for (int j = l * 4; j < (l + 1) * 4; j++) {
                shared_b[(lid_x >> 6 << 2) + (j & 3)][(lid_x & 63)] = pB[offset + j * WIDTH];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int k = 0; k < 4; k++) {
                sum += shared_a[(lid_x >> 6 << 6) + ((lid_x & 15) << 2) + k] *
                       shared_b[(lid_x >> 6 << 2) + ((lid_x & 63) >> 4)][((lid_x & 15) << 2) + k];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        result[l][lid_x] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        reduce(result[l], lid_x);
    }

    if (lid_x < 8) {
        c[(grid_x << 3) + lid_x] =
            result[lid_x >> 2][(lid_x & 3) << 4] + bias[(grid_x << 3) + lid_x];
    }
}
#elif KERNEL_METHOD == 4
#define ITER (WIDTH >> 6)

void reduce(__local float* buffer, int tid) {
    if (tid < 32) {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }

    if (tid < 16) {
        buffer[tid << 2] += buffer[(tid << 2) + 2];
    }
}

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c,
    uint N, uint WIDTH, uint OUTPUT) {
    __local float shared_a[2][66];
    __local float shared_b[8][66];
    __local float result[65];

    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __global const float* pA = (__global const float*)(a);
    __global const float* pB = (__global const float*)(b + (grid_x << 3) * WIDTH);

    int offset = ((grid_x << 6)) % WIDTH;
    float sum  = 0.0f;

    for (int i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
        for (int j = 0; j < 2; j++) {
            shared_a[j][lid_x + (lid_x >> 5)] = pA[offset + lid_x + j * WIDTH];
        }

        for (int j = 0; j < 8; j++) {
            shared_b[j][lid_x + (lid_x >> 5)] = pB[offset + lid_x + j * WIDTH];
        }

        for (int k = 0; k < 16; k++) {
            sum += shared_a[lid_x >> 5][((lid_x & 3) << 4) + k + ((((lid_x & 3) << 4) + k) >> 5)] *
                   shared_b[(lid_x & 31) >> 2]
                   [((lid_x & 3) << 4) + k + ((((lid_x & 3) << 4) + k) >> 5)];
        }
    }

    result[lid_x] = sum;
    reduce(result, lid_x);

    if (lid_x < 2) {
        float8 out;
        float* pOut = (float*)&out;

        for (int i = 0; i < 8; i++) {
            pOut[i] = result[((lid_x * 8 + i) << 2)] + bias[(grid_x << 3) + i];
        }

        __global float8* pC = (__global float8*)(c + (grid_x << 3) + lid_x * OUTPUT);
        *pC                 = out;
    }
}
#elif KERNEL_METHOD == 5
#define ITER (WIDTH >> 6)

void reduce(__local float* buffer, int tid) {
    if (tid < 32) {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }

    if (tid < 16) {
        buffer[tid << 2] += buffer[(tid << 2) + 2];
    }
}

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c,
    uint N, uint WIDTH, uint OUTPUT) {
    __local float shared_a[4][66];
    __local float shared_b[4][66];
    __local float result[65];

    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __global const float* pA = (__global const float*)(a);
    __global const float* pB = (__global const float*)(b + (grid_x << 2) * WIDTH);

    int offset = ((grid_x << 6)) % WIDTH;
    float sum  = 0.0f;

    for (int i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
        for (int j = 0; j < 4; j++) {
            shared_a[j][lid_x + (lid_x >> 5)] = pA[offset + lid_x + j * WIDTH];
            shared_b[j][lid_x + (lid_x >> 5)] = pB[offset + lid_x + j * WIDTH];
        }

        for (int k = 0; k < 16; k++) {
            sum += shared_a[lid_x >> 4][((lid_x & 3) << 4) + k + ((((lid_x & 3) << 4) + k) >> 5)] *
                   shared_b[(lid_x & 15) >> 2]
                   [((lid_x & 3) << 4) + k + ((((lid_x & 3) << 4) + k) >> 5)];
        }
    }

    result[lid_x] = sum;
    reduce(result, lid_x);

    if (lid_x < 4) {
        float4 out;
        float* pOut = (float*)&out;

        for (int i = 0; i < 4; i++) {
            pOut[i] = result[((lid_x * 4 + i) << 2)] + bias[(grid_x << 2) + i];
        }

        __global float4* pC = (__global float4*)(c + (grid_x << 2) + lid_x * OUTPUT);
        *pC                 = out;
    }
}
#elif KERNEL_METHOD == 6
#define ITER (WIDTH >> 6)

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c,
    uint N, uint WIDTH, uint OUTPUT) {
    __local float shared_a[8][66];
    __local float shared_b[8][66];
    __local float result[65];

    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __global const float* pA = (__global const float*)(a);
    __global const float* pB = (__global const float*)(b + (grid_x << 3) * WIDTH);

    int offset = ((grid_x << 6)) % WIDTH;
    float sum  = 0.0f;

    for (int i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
        for (int j = 0; j < 8; j++) {
            shared_a[j][lid_x + (lid_x >> 5)] = pA[offset + lid_x + j * WIDTH];
        }

        for (int j = 0; j < 8; j++) {
            shared_b[j][lid_x + (lid_x >> 5)] = pB[offset + lid_x + j * WIDTH];
        }

        for (int k = 0; k < 64; k++) {
            sum += shared_a[lid_x >> 3][k + (k >> 5)] * shared_b[(lid_x & 7)][k + (k >> 5)];
        }
    }

    result[lid_x] = sum;

    if (lid_x < 8) {
        float8 out;
        float* pOut = (float*)&out;

        for (int i = 0; i < 8; i++) {
            pOut[i] = result[((lid_x * 8 + i))] + bias[(grid_x << 3) + i];
        }

        __global float8* pC = (__global float8*)(c + (grid_x << 3) + lid_x * OUTPUT);
        *pC                 = out;
    }
}
#elif KERNEL_METHOD == 7
#define QSTRIDE (WIDTH >> 2)
#define ITER (WIDTH >> 7)

__attribute__((reqd_work_group_size(256, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c,
    uint N, uint WIDTH, uint OUTPUT) {
    __local float shared_a[1536];
    __local float shared_b[2560];
    __local float2* pShared_a = (__local float2*)shared_a;
    __local float4* pShared_b = (__local float4*)shared_b;

    float2 sha;
    float* pSha = (float*)&sha;
    float4 shb;
    float* pShb = (float*)&shb;
    float4 sum[2];
    float* pSum = (float*)sum;

    uint lid_x  = get_local_id(0);
    uint grid_x = get_group_id(0);

    __global const float* pA =
        (__global const float*)(a + (lid_x >> 6 << 3) * WIDTH + (grid_x >> 6) * QSTRIDE);
    __global const float* pB =
        (__global const float*)(b + (((grid_x & 63) << 6) + (lid_x >> 6 << 4)) * WIDTH +
                                (grid_x >> 6) * QSTRIDE);
    __global float4* pC    = (__global float4*)(c + ((lid_x >> 4 << 1) * OUTPUT +
                             ((grid_x & 63) << 6) + ((lid_x & 15) << 2)));
    __global float4* pBias = (__global float4*)(bias + ((grid_x & 63) << 6) + ((lid_x & 15) << 2));

    int offset = (((grid_x & 63) << 5)) % QSTRIDE;

    for (uint i = 0; i < 2; i++) {
        sum[i] = 0.0f;
    }

    for (ushort i = 0; i < ITER; i++, offset = (offset + 32) % QSTRIDE) {
        for (uint j = 0; j < 4; j++) {
            shared_a[((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 3)) + ((lid_x & 31)) * 32 +
                     ((((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 3)) +
                       ((lid_x & 31)) * 32) >>
                      5)] = pA[((j << 1) + ((lid_x & 63) >> 5)) * WIDTH + (lid_x & 31) + offset];
        }

        for (uint j = 0; j < 8; j++) {
            shared_b[((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 4)) + ((lid_x & 31)) * 64 +
                     ((((j << 1) + ((lid_x & 63) >> 5) + (lid_x >> 6 << 4)) +
                       ((lid_x & 31)) * 64) >>
                      5)] = pB[((j << 1) + ((lid_x & 63) >> 5)) * WIDTH + (lid_x & 31) + offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint k = 0; k < 32; k++) {
            for (uint m = 0; m < 2; m++) {
                pSha[m] = shared_a[((lid_x >> 4) << 1) + m + k * 32 +
                                   ((((lid_x >> 4) << 1) + m + k * 32) >> 5)];
            }

            for (uint l = 0; l < 4; l++) {
                pShb[l] = shared_b[((lid_x & 15) << 2) + l + k * 64 +
                                   ((((lid_x & 15) << 2) + l + k * 64) >> 5)];
            }

            for (uint m = 0; m < 2; m++) {
                for (uint l = 0; l < 4; l++) {
                    pSum[m * 4 + l] += pSha[m] * pShb[l];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if ((grid_x >> 6) == 0) {
        for (uint i = 0; i < 2; i++) {
            pC[i * OUTPUT >> 2] = sum[i] + pBias[0];
        }
    } else {
        for (uint i = 0; i < 2; i++) {
            pC[i * OUTPUT >> 2] += sum[i];
        }
    }
}
#elif KERNEL_METHOD == 8
#define ITER (WIDTH >> 6)
#define HWG (384 >> 1)

void reduce(__local float* buffer, int tid) {
    if (tid < 32) {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }
}

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c,
    uint N, uint WIDTH, uint OUTPUT) {
    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __local float shared_a[4][65];
    __local float shared_b[8][65];
    __local float result[65];

    __global const float* pA =
        (__global const float*)(a + ((grid_x / HWG) * (WIDTH << 2)));
    __global const float* pB = (__global const float*)(b);

    int offset = (((grid_x % HWG) << 6)) % WIDTH;

    float sum = 0.0f;

    for (int i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
        for (int j = 0; j < 4; j++) {
            shared_a[j][lid_x] = pA[offset + j * WIDTH + lid_x];
        }

        for (int j = 0; j < 8; j++) {
            shared_b[j][(lid_x)] =
                ((j + ((grid_x % HWG) << 3)) * WIDTH + (offset + lid_x) < OUTPUT * WIDTH
                 ? pB[(j + ((grid_x % HWG) << 3)) * WIDTH + (offset + lid_x)]
                 : 0.0f);
        }

        for (int k = 0; k < 32; k++) {
            sum += shared_a[lid_x >> 4][((lid_x & 1) << 5) + k] *
                   shared_b[((lid_x & 15) >> 1)][((lid_x & 1) << 5) + k];
        }
    }

    result[lid_x] = sum;
    reduce(result, lid_x);

    if (lid_x < 32 && ((grid_x % HWG) << 3) + (lid_x & 7) < OUTPUT) {
        int out_offset =
            ((grid_x / HWG << 2) + (lid_x >> 3)) * OUTPUT + ((grid_x % HWG) << 3) + (lid_x & 7);
        c[out_offset] = bias[((grid_x % HWG) << 3) + (lid_x & 7)] + result[(lid_x << 1)];
    }
}
#elif KERNEL_METHOD == 9
#define ITER (WIDTH >> 6)

void reduce(__local float* buffer, int tid) {
    if (tid < 32) {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }
}

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c,
    uint N, uint WIDTH, uint OUTPUT) {
    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __local float shared_a[4][65];
    __local float shared_b[8][65];
    __local float result[65];

    __global const float* pA = (__global const float*)(a + ((grid_x >> 7) * (WIDTH << 2)));
    __global const float* pB = (__global const float*)(b);

    int offset = (((grid_x & 127) << 6)) % WIDTH;

    float sum = 0.0f;

    for (int i = 0; i < ITER; i++, offset = (offset + 64) % WIDTH) {
        for (int j = 0; j < 4; j++) {
            shared_a[j][lid_x] = pA[offset + j * WIDTH + lid_x];
        }

        for (int j = 0; j < 8; j++) {
            shared_b[j][(lid_x)] =
                ((j + ((grid_x & 127) << 3)) * WIDTH + (offset + lid_x) < OUTPUT * WIDTH
                 ? pB[(j + ((grid_x & 127) << 3)) * WIDTH + (offset + lid_x)]
                 : 0.0f);
        }

        for (int k = 0; k < 32; k++) {
            sum += shared_a[lid_x >> 4][((lid_x & 1) << 5) + k] *
                   shared_b[((lid_x & 15) >> 1)][((lid_x & 1) << 5) + k];
        }
    }

    result[lid_x] = sum;
    reduce(result, lid_x);

    if (lid_x < 32 && ((grid_x & 127) << 3) + (lid_x & 7) < OUTPUT) {
        int out_offset =
            ((grid_x >> 7 << 2) + (lid_x >> 3)) * OUTPUT + ((grid_x & 127) << 3) + (lid_x & 7);
        c[out_offset] = bias[((grid_x & 127) << 3) + (lid_x & 7)] + result[(lid_x << 1)];
    }
}
#elif KERNEL_METHOD == 10
#ifndef ATOMIC
#define ATOMIC  32
#endif

#ifndef LOCAL_SIZE
#define LOCAL_SIZE  64
#endif

#define LOCAL_MEMORY    (LOCAL_SIZE + (LOCAL_SIZE >> 5))

#define WORKLOAD    (LOCAL_SIZE * ATOMIC)
#define ITER ((WIDTH + WORKLOAD - 1) / WORKLOAD)


__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b,
#ifdef BIAS
    __global const float* bias,
#endif
    __global float* c,
    __global float* gAtomicLock,
    uint N, uint WIDTH, uint OUTPUT) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);
    uint col_id = grid_x / ATOMIC;
    uint atomic_id = grid_x % ATOMIC;

    __local float result[2][LOCAL_MEMORY];

    __global const float* pB;
    __global float* pC;

    volatile __global uint* pCounter = (volatile __global uint*)(gAtomicLock);

    uint wave_stride = ITER * WORKLOAD / (WORKLOAD >> 6);
    uint offset = (grid_x >> 2 << 6) % wave_stride;
    uint wave_id = (atomic_id << 4) + (lid_x >> 6);

    float sum = 0.0f;

    float previous_value;
    uint prevVal;
    uint newVal;

    if (grid_x % ATOMIC == 0) {
        for (uint n = 0; n < N; n++) {
            if (lid_x == 0) {
                *(pCounter + col_id + n * OUTPUT) = 0;
            }
        }
    }

    if (col_id < OUTPUT) {
        for (uint n = 0; n < N; n++) {

            pC = (__global float*)(c + col_id);

            sum = 0.0f;

            for (uint i = 0; i < ITER; i++, offset = (offset + 64) % wave_stride) {
                //result[n % 2][lid_x + (lid_x >> 5)] = (offset + wave_id * wave_stride + (lid_x & 63) < WIDTH ? a[offset + wave_id * wave_stride + (lid_x & 63) + n * WIDTH] : 0.0f);
                if ((offset + wave_id * wave_stride + (lid_x & 63) + n * WIDTH < N * WIDTH) &&
                        (col_id * WIDTH + offset + wave_id * wave_stride + (lid_x & 63)) < WIDTH * OUTPUT) {
                    result[n % 2][lid_x + (lid_x >> 5)] = a[offset + wave_id * wave_stride + (lid_x & 63) + n * WIDTH];

                    pB = (__global const float*)(b + col_id * WIDTH + offset + wave_id * wave_stride + (lid_x & 63));

                    sum += (offset + wave_id * wave_stride + (lid_x & 63) < WIDTH ?
                            result[n % 2][lid_x + (lid_x >> 5)] * pB[0] : 0.0f);
                }
            }

            result[n % 2][lid_x + (lid_x >> 5)] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint i = LOCAL_SIZE >> 1; i >= 64; i >>= 1) {
                if (lid_x < i) {
                    result[n % 2][lid_x + (lid_x >> 5)] += result[n % 2][lid_x + i + ((lid_x + i) >> 5)];
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid_x < 32) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 33];
            }

            if (lid_x < 16) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 16];
            }

            if (lid_x < 8) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 8];
            }

            if (lid_x < 4) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 4];
            }

            if (lid_x < 2) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 2];
            }

            if (lid_x < 1) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 1];
            }

            if (grid_x % ATOMIC == 0) {
#ifdef BIAS
                pC[n * OUTPUT] = bias[col_id] + result[n % 2][0];
#else
                pC[n * OUTPUT] = result[n % 2][0];
#endif

                barrier(CLK_GLOBAL_MEM_FENCE);

                if (lid_x == 0) {
                    atomic_inc(pCounter + col_id + n * OUTPUT);
                }
            } else {
                if (lid_x == 0) {
                    do {
                    } while (atomic_cmpxchg((volatile __global uint*)(pCounter + col_id + n * OUTPUT), 1, 1) == 0);
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                if (lid_x == 0) {
                    do {
                        previous_value = pC[n * OUTPUT];
                        prevVal = as_uint(previous_value);
                        newVal = as_uint(result[n % 2][0] + previous_value);
                    } while (atomic_cmpxchg((__global uint*)(pC + n * OUTPUT), prevVal, newVal) != prevVal);
                }
            }
        }
    }
}
#elif KERNEL_METHOD == 11
#ifndef LOCAL_SIZE
#define LOCAL_SIZE  64
#endif

#define LOCAL_MEMORY    (LOCAL_SIZE + (LOCAL_SIZE >> 5))

#define ITER ((WIDTH + LOCAL_SIZE - 1) / LOCAL_SIZE)

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b,
#ifdef BIAS
    __global const float* bias,
#endif
    __global float* c, uint N, uint WIDTH, uint OUTPUT) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float result[2][LOCAL_MEMORY];

    __global const float* pB; // correct
    __global float* pC;

    uint wave_stride = ITER * LOCAL_SIZE / (LOCAL_SIZE >> 6);
    uint offset = (grid_x >> 2 << 6) % wave_stride;
    uint wave_id = (lid_x >> 6);

    float sum = 0.0f;

    if (grid_x < OUTPUT) {
        for (uint n = 0; n < N; n++) {
            sum = 0.0f;

            for (uint i = 0; i < ITER; i++, offset = (offset + 64) % wave_stride) {
                //result[n % 2][lid_x + (lid_x >> 5)] = (offset + wave_id * wave_stride + (lid_x & 63) < WIDTH ? a[offset + wave_id * wave_stride + (lid_x & 63) + n * WIDTH] : 0.0f);
                result[n % 2][lid_x + (lid_x >> 5)] = a[offset + wave_id * wave_stride + (lid_x & 63) + n * WIDTH];

                pB = (__global const float*)(b + grid_x * WIDTH + offset + wave_id * wave_stride + (lid_x & 63));
                pC = (__global float*)(c + grid_x);

                sum += (offset + wave_id * wave_stride + (lid_x & 63) < WIDTH ?
                        result[n % 2][lid_x + (lid_x >> 5)] * pB[0] : 0.0f);
            }

            result[n % 2][lid_x + (lid_x >> 5)] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint i = LOCAL_SIZE >> 1; i >= 64; i >>= 1) {
                if (lid_x < i) {
                    result[n % 2][lid_x + (lid_x >> 5)] += result[n % 2][lid_x + i + ((lid_x + i) >> 5)];
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }


            if (lid_x < 32) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 33];
            }

            if (lid_x < 16) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 16];
            }

            if (lid_x < 8) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 8];
            }

            if (lid_x < 4) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 4];
            }

            if (lid_x < 2) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 2];
            }

            if (lid_x < 1) {
                result[n % 2][lid_x] += result[n % 2][lid_x + 1];
            }

            if (lid_x == 0) {
#ifdef BIAS
                pC[n * OUTPUT] = bias[grid_x] + result[n % 2][0];
#else
                pC[n * OUTPUT] = result[n % 2][0];
#endif
            }
        }
    }
}
#endif
#endif
