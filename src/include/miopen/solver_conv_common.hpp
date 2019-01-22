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

#ifndef SOLVER_CONV_COMMON_H
#define SOLVER_CONV_COMMON_H

#include "miopen/solver.hpp"
#include <iostream>
#include <vector>
#define ALOGD(X) std::cout << X << std::endl;
#define ALOGE(X) std::cerr << X << std::endl;

namespace miopen {

enum DeviceType { GFX803, GFX900 };

struct Conv11Param {
    int global_split;
    int stride_per_iter;
    int tile_col;
    int tile_row;
    int wi_per_tile_row;
    int wi_per_tile_col;
    int code_branch;
    int code_method;
};

struct Conv1x1Type {
    int dev;
    int batch;
    int stride;
    int height;
    int width;
    int channel;
    int output_num;
    int kernel_method;
    bool fusion_pooling;
    std::string kernel_name;
    Conv11Param params;
};

class ConvCommon {
public:
    ConvCommon() {}
    ~ConvCommon() {}

    bool getKernelInfo(const ConvolutionContext& params, Conv1x1Type& mType);

    bool _usemacro = false;

private:
    std::vector<Conv1x1Type> conv1x1type {

        //*********************************************************
        // GFX803
        //*********************************************************
        //---------------------------------------------
        // stride = 1
        //---------------------------------------------

        // tensile for Resnet50, Resnet101
        //{GFX803, 1, 1, 7, 7, 512, 2048, 0, "ConvFwd1x1_7x7x512x2048x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        //{GFX803, 1, 1, 7, 7, 2048, 512, 0, "ConvFwd1x1_7x7x2048x512x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 14, 14, 256, 1024, 0, false, "ConvFwd1x1_14x14x256x1024x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 14, 14, 1024, 256, 0, false, "ConvFwd1x1_14x14x1024x256x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 28, 28, 128, 512, 0, false, "ConvFwd1x1_28x28x128x512x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 28, 28, 512, 128, 0, false, "ConvFwd1x1_28x28x512x128x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 56, 56, 64, 64, 0, false, "ConvFwd1x1_56x56x64x64x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 56, 56, 64, 256, 0, false, "ConvFwd1x1_56x56x64x256x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 56, 56, 256, 64, 0, false, "ConvFwd1x1_56x56x256x64x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        //{GFX803, 2, 1, 7, 7, 512, 2048, 0, "ConvFwd1x1_7x7x512x2048x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        //{GFX803, 2, 1, 7, 7, 2048, 512, 0, "ConvFwd1x1_7x7x2048x512x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 14, 14, 256, 1024, 0, false, "ConvFwd1x1_14x14x256x1024x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 14, 14, 1024, 256, 0, false, "ConvFwd1x1_14x14x1024x256x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 28, 28, 128, 512, 0, false, "ConvFwd1x1_28x28x128x512x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 28, 28, 512, 128, 0, false, "ConvFwd1x1_28x28x512x128x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 56, 56, 64, 64, 0, false, "ConvFwd1x1_56x56x64x64x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 56, 56, 64, 256, 0, false, "ConvFwd1x1_56x56x64x256x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 56, 56, 256, 64, 0, false, "ConvFwd1x1_56x56x256x64x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 7, 7, 512, 2048, 0, false, "ConvFwd1x1_7x7x512x2048x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 7, 7, 2048, 512, 0, false, "ConvFwd1x1_7x7x2048x512x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 14, 14, 256, 1024, 0, false, "ConvFwd1x1_14x14x256x1024x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 14, 14, 1024, 256, 0, false, "ConvFwd1x1_14x14x1024x256x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 28, 28, 128, 512, 0, false, "ConvFwd1x1_28x28x128x512x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 28, 28, 512, 128, 0, false, "ConvFwd1x1_28x28x512x128x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 56, 56, 64, 64, 0, false, "ConvFwd1x1_56x56x64x64x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 56, 56, 64, 256, 0, false, "ConvFwd1x1_56x56x64x256x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 56, 56, 256, 64, 0, false, "ConvFwd1x1_56x56x256x64x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},

        // width = 1
        // Mobilenet, W1C1024K1000 *max 8

        {GFX803, 8, 1, 1, 1, 1024, 1000, 2, false, "Conv1x1FC.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // MobilenetV2, W1C1280K1000 * max 8
        {GFX803, 8, 1, 1, 1, 1280, 1000, 2, false, "Conv1x1FC.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // width = 7
        // MobilenetV2, W7C160K960
        {GFX803, 1, 1, 7, 7, 160, 960, 1, false, "Conv1x1.cl", {2, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7C160K960 *2
        {GFX803, 2, 1, 7, 7, 160, 960, 1, false, "Conv1x1.cl", {2, 16, 16, 16, 1, 1, 3, 1}},
        // MobilenetV2, W7C320K1280POOL
        {GFX803, 1, 1, 7, 7, 320, 1280, 3, true, "Conv1x1.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // MobilenetV2, W7C320K1280POOL *2
        {GFX803, 1, 1, 7, 7, 320, 1280, 3, true, "Conv1x1.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Mobilenet, W7C512K1024
        {GFX803, 2, 1, 7, 7, 512, 1024, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 3, 1}},
        // Resnet, W7C512K2048
        {GFX803, 1, 1, 7, 7, 512, 2048, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W7C512K2048 *2
        {GFX803, 2, 1, 7, 7, 512, 2048, 1, false, "Conv1x1.cl", {2, 32, 32, 32, 2, 2, 3, 1}},
        // MobilenetV2, W7CXK160
        {GFX803, 1, 1, 7, 7, 576, 160, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK160 *2
        {GFX803, 2, 1, 7, 7, 576, 160, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 3, 1}},
        // MobilenetV2, W7CXK160
        {GFX803, 1, 1, 7, 7, 960, 160, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK160 *2
        {GFX803, 2, 1, 7, 7, 960, 160, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 3, 1}},
        // MobilenetV2, W7CXK320
        {GFX803, 1, 1, 7, 7, 960, 320, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK320 *2
        {GFX803, 2, 1, 7, 7, 960, 320, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 3, 1}},
        // Mobilenet, W7C1024K1024POOL
        {GFX803, 1, 1, 7, 7, 1024, 1024, 2, true, "Conv1x1.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Mobilenet, W7C1024K1024POOL *2
        {GFX803, 2, 1, 7, 7, 1024, 1024, 2, true, "Conv1x1.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Resnet, W7C2048K512
        {GFX803, 1, 1, 7, 7, 2048, 512, 1, false, "Conv1x1.cl", {8, 32, 32, 32, 2, 2, 1, 1}},
        // Resnet, W7C2048K512 *2
        {GFX803, 2, 1, 7, 7, 2048, 512, 1, false, "Conv1x1.cl", {8, 32, 32, 32, 2, 2, 3, 1}},
        // width = 14
        // MobilenetV2, W14C96K576
        {GFX803, 1, 1, 14, 14, 96, 576, 1, false, "Conv1x1.cl", {2, 16, 32, 32, 4, 1, 1, 1}},
        // MobilenetV2, W14C96K576 *2
        {GFX803, 2, 1, 14, 14, 96, 576, 1, false, "Conv1x1.cl", {2, 16, 32, 32, 4, 1, 3, 1}},
        // Mobilenet, W14C256K512
        {GFX803, 2, 1, 14, 14, 256, 512, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 3, 1}},
        // MobilenetV2, W14CXK96
        {GFX803, 1, 1, 14, 14, 384, 96, 1, false, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // MobilenetV2, W14CXK96 *2
        {GFX803, 2, 1, 14, 14, 384, 96, 1, false, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 3, 1}},
        // Mobilenet, W14C512K512
        {GFX803, 2, 1, 14, 14, 512, 512, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 3, 1}},
        // MobilenetV2, W14CXK96
        {GFX803, 1, 1, 14, 14, 576, 96, 1, false, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // MobilenetV2, W14CXK96 *2
        {GFX803, 2, 1, 14, 14, 576, 96, 1, false, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 3, 1}},
        // YOLO, W14C1024K512
        {GFX803, 1, 1, 14, 14, 1024, 512, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // YOLO, W14C1024K512 *2
        {GFX803, 2, 1, 14, 14, 1024, 512, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 3, 1}},

        // width = 28
        // MobilenetV2, C32H28W28K192S1 but not implement

        // MobilenetV2, C64H28W28K384S1 but not implement

        // Mobilenet, C128H28W28K256S1 but not implement

        // Resnet, C128H28W28K512S1 but not implement

        // MobilenetV2, W28C144K32
        {GFX803, 2, 1, 28, 28, 144, 32, 1, false, "Conv1x1.cl", {1, 16, 16, 16, 1, 1, 3, 1}},
        // MobilenetV2, W28C192K32
        {GFX803, 2, 1, 28, 28, 192, 32, 1, false, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 3, 1}},
        // MobilenetV2, W28C192K64
        {GFX803, 2, 1, 28, 28, 192, 64, 1, false, "Conv1x1.cl", {4, 16, 16, 64, 2, 2, 3, 1}},
        // Mobilenet, W28C256K256
        {GFX803, 1, 1, 28, 28, 256, 256, 1, false, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // Mobilenet, W28C256K256 *2
        {GFX803, 2, 1, 28, 28, 256, 256, 1, false, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // MobilenetV2, W28C384K64
        {GFX803, 2, 1, 28, 28, 384, 64, 1, false, "Conv1x1.cl", {8, 16, 32, 64, 4, 2, 3, 1}},
        // YOLO, W28C512K256
        {GFX803, 1, 1, 28, 28, 512, 256, 1, false, "Conv1x1.cl", {2, 32, 32, 64, 4, 2, 1, 1}},
        // YOLO, W28C512K256 *2
        {GFX803, 2, 1, 28, 28, 512, 256, 1, false, "Conv1x1.cl", {2, 32, 32, 64, 4, 2, 3, 1}},
        // YOLO, C512H28W28K512S1 but not implement

        // width =56
        // MobilenetV2, C24H56W56K144S1 but not implement

        // Resnet, C64H56W56K64S1 but not implement

        // Mobilenet, C64H56W56K128S1 but not implement

        // Resnet, C64H56W56K256S1 but not implement

        // MobilenetV2, C96H56W56K24S1 but not implement

        // Mobilenet, C128H56W56K128S1 but not implement

        // MobilenetV2, C144H56W56K24S1 but not implement

        // YOLO, C192H56W56K128S1 but not implement

        // YOLO, C256H56W56K256S1 but not implement

        // width = 112
        // MobilenetV2, C16H112W112K96S1 but not implement

        // MobilenetV2, C32H112W112K16S1 but not implement

        // MobilenetV2, C32H112W112K32S1 but not implement

        // Mobilenet, C32H112W112K64S1

        //---------------------------------------------
        // stride = 2
        //---------------------------------------------
        // width = 14
        // Resnet, W14C1024K512S2
        {GFX803, 1, 2, 14, 14, 1024, 512, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C1024K512S2 *2
        {GFX803, 2, 2, 14, 14, 1024, 512, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 3, 1}},
        // Resnet, W14C1024K2048S2
        {GFX803, 1, 2, 14, 14, 1024, 2048, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C1024K2048S2
        {GFX803, 2, 2, 14, 14, 1024, 2048, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 3, 1}},
        // width = 28
        // Resnet, W28C512K256S2
        {GFX803, 1, 2, 28, 28, 512, 256, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W28C512K256S2 *2
        {GFX803, 2, 2, 28, 28, 512, 256, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 3, 1}},
        // Resnet, W28C512K1024S2
        {GFX803, 1, 2, 28, 28, 512, 1024, 1, false, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // Resnet, W28C512K1024S2 *2
        {GFX803, 2, 2, 28, 28, 512, 1024, 1, false, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 3, 1}},
        // width = 56
        // Resnet, W56C256K128S2
        {GFX803, 1, 2, 56, 56, 256, 128, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W56C256K128S2 *2
        {GFX803, 2, 2, 56, 56, 256, 128, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 3, 1}},
        // Resnet, W56C256K512S2
        {GFX803, 1, 2, 56, 56, 256, 512, 1, false, "Conv1x1.cl", {1, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W56C256K512S2 *2
        {GFX803, 2, 2, 56, 56, 256, 512, 1, false, "Conv1x1.cl", {1, 32, 32, 64, 4, 2, 3, 1}},


        // Inception
        // W56C96K128
        {GFX803, 1, 1, 56, 56, 96, 128, 1, false, "Conv1x1.cl", { 2, 16, 64, 32, 4, 2, 1, 1 }},
        // W28CXK96
        {GFX803, 1, 1, 28, 28, 0, 96, 1, false, "Conv1x1.cl", { 6, 16, 16, 32, 2, 1, 1, 1 }},
        // W28C288K48
        {GFX803, 1, 1, 28, 28, 288, 48, 1, false, "Conv1x1.cl", { 6, 16, 32, 16, 2, 1, 1, 1 }},
        // W28C480K192
        {GFX803, 1, 1, 28, 28, 480, 192, 1, false, "Conv1x1.cl", { 4, 16, 64, 32, 4, 2, 1, 1 }},
        // W14C864K224
        {GFX803, 1, 1, 14, 14, 864, 224, 1, false, "Conv1x1.cl", { 6, 16, 32, 32, 2, 2, 1, 1 }},
        // W14C864K64
        {GFX803, 1, 1, 14, 14, 864, 64, 1, false, "Conv1x1.cl", { 9, 16, 16, 16, 1, 1, 1, 1 }},
        // W14CXK128
        {GFX803, 1, 1, 14, 14, 0, 128, 1, false, "Conv1x1.cl", { 4, 16, 16, 32, 2, 1, 1, 1 }},
        // W14CXK192
        {GFX803, 1, 1, 14, 14, 0, 192, 1, false, "Conv1x1.cl", { 4, 16, 16, 32, 2, 1, 1, 1 }},
        // W14CXK160
        {GFX803, 1, 1, 14, 14, 0, 160, 1, false, "Conv1x1.cl", { 9, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK352
        {GFX803, 1, 1, 7, 7, 0, 352, 1, false, "Conv1x1.cl", { 8, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK192
        {GFX803, 1, 1, 7, 7, 0, 192, 1, false, "Conv1x1.cl", { 12, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK128
        {GFX803, 1, 1, 7, 7, 0, 128, 1, false, "Conv1x1.cl", { 12, 16, 16, 32, 2, 1, 1, 1 }},

        // VGG-SSD
        // W19C1024K1024
        {GFX803, 1, 1, 19, 19, 1024, 1024, 1, false, "Conv1x1.cl", { 4, 16, 64, 64, 8, 2, 1, 1 }},
        // W19C1024K256
        {GFX803, 1, 1, 19, 19, 1024, 256, 1, false, "Conv1x1.cl", { 4, 16, 64, 32, 4, 2, 1, 1 }},
        // W10C512K128
        {GFX803, 1, 1, 10, 10, 512, 128, 1, false, "Conv1x1.cl", { 8, 16, 32, 16, 2, 1, 1, 1 }},
        // W5C256K128
        {GFX803, 1, 1, 5, 5, 256, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},
        // W3C256K128
        {GFX803, 1, 1, 3, 3, 256, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},

        // Mobilenet-SSD
        // W150C32K64
        {GFX803, 1, 1, 150, 150, 32, 64, 1, false, "Conv1x1.cl", { 1, 16, 64, 64, 8, 2, 1, 1 }},
        // W75CXK128
        {GFX803, 1, 1, 75, 75, 0, 128, 1, false, "Conv1x1.cl", { 1, 16, 64, 64, 8, 2, 1, 1 }},
        // W38CXK256
        {GFX803, 1, 1, 38, 38, 0, 256, 1, false, "Conv1x1.cl", { 2, 16, 64, 64, 8, 2, 1, 1 }},
        // W19CXK512
        {GFX803, 1, 1, 19, 19, 0, 512, 1, false, "Conv1x1.cl", { 4, 16, 64, 64, 8, 2, 1, 1 }},
        // W10CXK1024
        {GFX803, 1, 1, 10, 10, 0, 1024, 1, false, "Conv1x1.cl", { 8, 16, 64, 64, 8, 2, 1, 1 }},
        // W10C1024K256
        {GFX803, 1, 1, 10, 10, 1024, 256, 1, false, "Conv1x1.cl", { 8, 16, 32, 32, 2, 2, 1, 1 }},
        // W5C512K128
        {GFX803, 1, 1, 5, 5, 512, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},
        // W2C256K64
        {GFX803, 1, 1, 2, 2, 256, 64, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},

        // Inception
        // W56C96K128
        {GFX803, 2, 1, 56, 56, 96, 128, 1, false, "Conv1x1.cl", { 2, 16, 64, 32, 4, 2, 1, 1 }},
        // W28CXK96
        {GFX803, 2, 1, 28, 28, 0, 96, 1, false, "Conv1x1.cl", { 6, 16, 16, 32, 2, 1, 1, 1 }},
        // W28C288K48
        {GFX803, 2, 1, 28, 28, 288, 48, 1, false, "Conv1x1.cl", { 6, 16, 32, 16, 2, 1, 1, 1 }},
        // W28C480K192
        {GFX803, 2, 1, 28, 28, 480, 192, 1, false, "Conv1x1.cl", { 4, 16, 64, 32, 4, 2, 1, 1 }},
        // W14C864K224
        {GFX803, 2, 1, 14, 14, 864, 224, 1, false, "Conv1x1.cl", { 6, 16, 32, 32, 2, 2, 1, 1 }},
        // W14C864K64
        {GFX803, 2, 1, 14, 14, 864, 64, 1, false, "Conv1x1.cl", { 9, 16, 16, 16, 1, 1, 1, 1 }},
        // W14CXK128
        {GFX803, 2, 1, 14, 14, 0, 128, 1, false, "Conv1x1.cl", { 9, 16, 16, 32, 2, 1, 1, 1 }},
        // W14CXK192
        {GFX803, 2, 1, 14, 14, 0, 192, 1, false, "Conv1x1.cl", { 9, 16, 16, 32, 2, 1, 1, 1 }},
        // W14CXK160
        {GFX803, 2, 1, 14, 14, 0, 160, 1, false, "Conv1x1.cl", { 9, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK352
        {GFX803, 2, 1, 7, 7, 0, 352, 1, false, "Conv1x1.cl", { 8, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK192
        {GFX803, 2, 1, 7, 7, 0, 192, 1, false, "Conv1x1.cl", { 12, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK128
        {GFX803, 2, 1, 7, 7, 0, 128, 1, false, "Conv1x1.cl", { 12, 16, 16, 32, 2, 1, 1, 1 }},

        // VGG-SSD
        // W19C1024K1024
        {GFX803, 2, 1, 19, 19, 1024, 1024, 1, false, "Conv1x1.cl", { 4, 16, 64, 64, 8, 2, 1, 1 }},
        // W19C1024K256
        {GFX803, 2, 1, 19, 19, 1024, 256, 1, false, "Conv1x1.cl", { 4, 16, 64, 32, 4, 2, 1, 1 }},
        // W10C512K128
        {GFX803, 2, 1, 10, 10, 512, 128, 1, false, "Conv1x1.cl", { 8, 16, 32, 16, 2, 1, 1, 1 }},
        // W5C256K128
        {GFX803, 2, 1, 5, 5, 256, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},
        // W3C256K128
        {GFX803, 2, 1, 3, 3, 256, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},

        // Mobilenet-SSD
        // W150C32K64
        {GFX803, 2, 1, 150, 150, 32, 64, 1, false, "Conv1x1.cl", { 1, 16, 64, 64, 8, 2, 1, 1 }},
        // W75CXK128
        {GFX803, 2, 1, 75, 75, 0, 128, 1, false, "Conv1x1.cl", { 1, 16, 64, 64, 8, 2, 1, 1 }},
        // W38CXK256
        {GFX803, 2, 1, 38, 38, 0, 256, 1, false, "Conv1x1.cl", { 2, 16, 64, 64, 8, 2, 1, 1 }},
        // W19CXK512
        {GFX803, 2, 1, 19, 19, 0, 512, 1, false, "Conv1x1.cl", { 4, 16, 64, 64, 8, 2, 1, 1 }},
        // W10CXK1024
        {GFX803, 2, 1, 10, 10, 0, 1024, 1, false, "Conv1x1.cl", { 8, 16, 64, 64, 8, 2, 1, 1 }},
        // W10C1024K256
        {GFX803, 2, 1, 10, 10, 1024, 256, 1, false, "Conv1x1.cl", { 8, 16, 32, 32, 2, 2, 1, 1 }},
        // W5C512K128
        {GFX803, 2, 1, 5, 5, 512, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},
        // W2C256K64
        {GFX803, 2, 1, 2, 2, 256, 64, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},


        // Wildcat
        {GFX803, 1, 0, 0, 0, 0, 0, 4, false, "Conv1x1.cl", { 0, 0, 0, 0, 0, 0, 0, 0 }},
        // Wildcat
        {GFX803, 2, 0, 0, 0, 0, 0, 4, false, "Conv1x1.cl", { 0, 0, 0, 0, 0, 0, 0, 0 }},


        //*********************************************************
        // GFX900
        //*********************************************************
        //---------------------------------------------
        // stride = 1
        //---------------------------------------------

        // width = 1
        // Mobilenet, W1C1024K1000 *max 8
        {GFX900, 8, 1, 1, 1, 1024, 1000, 2, false, "Conv1x1FC.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // MobilenetV2, W1C1280K1000 *max 8
        {GFX900, 8, 1, 1, 1, 1280, 1000, 2, false, "Conv1x1FC.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // width = 7
        // MobilenetV2, W7C160K960
        {GFX900, 1, 1, 7, 7, 160, 960, 1, false, "Conv1x1.cl", {2, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7C160K960 *2
        {GFX900, 2, 1, 7, 7, 160, 960, 1, false, "Conv1x1.cl", {2, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7C320K1280POOL
        {GFX900, 1, 1, 7, 7, 320, 1280, 3, true, "Conv1x1.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // MobilenetV2, W7C320K1280POOL *2
        {GFX900, 2, 1, 7, 7, 320, 1280, 3, true, "Conv1x1.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Mobilenet, W7C512K1024
        {GFX900, 2, 1, 7, 7, 512, 1024, 1, false, "Conv1x1.cl", {4, 32, 32, 32, 4, 1, 1, 1}},
        // Resnet, W7C512K2048
        {GFX900, 1, 1, 7, 7, 512, 2048, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W7C512K2048 *2
        {GFX900, 2, 1, 7, 7, 512, 2048, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // MobilenetV2, W7CXK160
        {GFX900, 1, 1, 7, 7, 576, 160, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK160 *2
        {GFX900, 2, 1, 7, 7, 576, 160, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK160
        {GFX900, 1, 1, 7, 7, 960, 160, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK160 *2
        {GFX900, 2, 1, 7, 7, 960, 160, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK320
        {GFX900, 1, 1, 7, 7, 960, 320, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK320 *2
        {GFX900, 2, 1, 7, 7, 960, 320, 1, false, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // Mobilenet, W7C1024K1024POOL
        {GFX900, 1, 1, 7, 7, 1024, 1024, 2, true, "Conv1x1.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Mobilenet, W7C1024K1024POOL *2
        {GFX900, 2, 1, 7, 7, 1024, 1024, 2, true, "Conv1x1.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Resnet, W7C2048K512
        {GFX900, 1, 1, 7, 7, 2048, 512, 1, false, "Conv1x1.cl", {8, 32, 32, 32, 4, 1, 1, 1}},
        // Resnet, W7C2048K512 *2
        {GFX900, 2, 1, 7, 7, 2048, 512, 1, false, "Conv1x1.cl", {8, 32, 32, 32, 4, 1, 1, 1}},
        // width = 14
        // MobilenetV2, W14C96K576
        {GFX900, 1, 1, 14, 14, 96, 576, 1, false, "Conv1x1.cl", {2, 16, 32, 32, 4, 1, 1, 1}},
        // MobilenetV2, W14C96K576 *2
        {GFX900, 2, 1, 14, 14, 96, 576, 1, false, "Conv1x1.cl", {2, 16, 32, 32, 4, 1, 1, 1}},
        // Mobilenet, W14C256K512
        {GFX900, 2, 1, 14, 14, 256, 512, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C256K1024
        {GFX900, 2, 1, 14, 14, 256, 1024, 1, false, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // MobilenetV2, W14CXK96
        {GFX900, 1, 1, 14, 14, 384, 96, 1, false, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // MobilenetV2, W14CXK96 *2
        {GFX900, 2, 1, 14, 14, 384, 96, 1, false, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // Mobilenet, W14C512K512
        {GFX900, 2, 1, 14, 14, 512, 512, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // MobilenetV2, W14CXK96
        {GFX900, 1, 1, 14, 14, 576, 96, 1, false, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // MobilenetV2, W14CXK96 *2
        {GFX900, 2, 1, 14, 14, 576, 96, 1, false, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // Resnet, W14C1024K256
        {GFX900, 1, 1, 14, 14, 1024, 256, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C1024K256 *2
        {GFX900, 2, 1, 14, 14, 1024, 256, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // YOLO, W14C1024K512
        {GFX900, 1, 1, 14, 14, 1024, 512, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // YOLO,  W14C1024K512 *2
        {GFX900, 2, 1, 14, 14, 1024, 512, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},

        // width = 28
        // MobilenetV2, C32H28W28K192S1 but not implement

        // MobilenetV2, C64H28W28K384S1 but not implement

        // Mobilenet, C128H28W28K256S1 but not implement

        // Resnet, C128H28W28K512S1 but not implement

        // MobilenetV2, W28C144K32
        {GFX900, 2, 1, 28, 28, 144, 32, 1, false, "Conv1x1.cl", {1, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W28C192K32
        {GFX900, 2, 1, 28, 28, 192, 32, 1, false, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // MobilenetV2, W28C192K64
        {GFX900, 2, 1, 28, 28, 192, 64, 1, false, "Conv1x1.cl", {4, 16, 16, 64, 4, 1, 1, 1}},
        // Mobilenet, W28C256K256
        {GFX900, 1, 1, 28, 28, 256, 256, 1, false, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // Mobilenet, W28C256K256 *2
        {GFX900, 2, 1, 28, 28, 256, 256, 1, false, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // MobilenetV2, W28C384K64
        {GFX900, 2, 1, 28, 28, 384, 64, 1, false, "Conv1x1.cl", {4, 16, 32, 32, 4, 1, 1, 1}},
        // Resnet, W28C512K128
        {GFX900, 2, 1, 28, 28, 512, 128, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // YOLO, W28C512K256
        {GFX900, 1, 1, 28, 28, 512, 256, 1, false, "Conv1x1.cl", {2, 32, 32, 64, 4, 2, 1, 1}},
        // YOLO, W28C512K256 *2
        {GFX900, 2, 1, 28, 28, 512, 256, 1, false, "Conv1x1.cl", {2, 32, 32, 64, 4, 2, 1, 1}},
        // YOLO, C512H28W28K512S1 but not implement

        // width =56
        // MobilenetV2, C24H56W56K144S1 but not implement

        // Resnet, C64H56W56K64S1 but not implement

        // Mobilenet, C64H56W56K128S1 but not implement

        // Resnet, C64H56W56K256S1 but not implement

        // MobilenetV2, C96H56W56K24S1 but not implement

        // Mobilenet, C128H56W56K128S1 but not implement

        // MobilenetV2, C144H56W56K24S1 but not implement

        // YOLO, C192H56W56K128S1 but not implement

        // Resnet, W56C256K64
        {GFX900, 2, 1, 56, 56, 256, 64, 1, false, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // YOLO, C256H56W56K256S1 but not implement

        // width = 112
        // MobilenetV2, C16H112W112K96S1 but not implement

        // MobilenetV2, C32H112W112K16S1 but not implement

        // MobilenetV2, C32H112W112K32S1 but not implement

        // Mobilenet, C32H112W112K64S1

        //---------------------------------------------
        // stride = 2
        //---------------------------------------------
        // width = 14
        // Resnet, W14C1024K512S2
        {GFX900, 1, 2, 14, 14, 1024, 512, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C1024K512S2 *2
        {GFX900, 2, 2, 14, 14, 1024, 512, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C1024K2048S2
        {GFX900, 1, 2, 14, 14, 1024, 2048, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C1024K2048S2 *2
        {GFX900, 2, 2, 14, 14, 1024, 2048, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // width = 28
        // Resnet, W28C512K256S2
        {GFX900, 1, 2, 28, 28, 512, 256, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W28C512K256S2 *2
        {GFX900, 2, 2, 28, 28, 512, 256, 1, false, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 3, 1}},
        // Resnet, W28C512K1024S2
        {GFX900, 1, 2, 28, 28, 512, 1024, 1, false, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // Resnet, W28C512K1024S2 *2
        {GFX900, 2, 2, 28, 28, 512, 1024, 1, false, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // width = 56
        // Resnet, W56C256K128S2
        {GFX900, 1, 2, 56, 56, 256, 128, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W56C256K128S2 *2
        {GFX900, 2, 2, 56, 56, 256, 128, 1, false, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W56C256K512S2
        {GFX900, 1, 2, 56, 56, 256, 512, 1, false, "Conv1x1.cl", {1, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W56C256K512S2 *2
        {GFX900, 2, 2, 56, 56, 256, 512, 1, false, "Conv1x1.cl", {1, 32, 32, 64, 4, 2, 1, 1}},

        // Inception
        // W56C96K128
        {GFX900, 1, 1, 56, 56, 96, 128, 1, false, "Conv1x1.cl", { 2, 16, 64, 32, 4, 2, 1, 1 }},
        // W28CXK96
        {GFX900, 1, 1, 28, 28, 0, 96, 1, false, "Conv1x1.cl", { 6, 16, 16, 32, 2, 1, 1, 1 }},
        // W28C288K48
        {GFX900, 1, 1, 28, 28, 288, 48, 1, false, "Conv1x1.cl", { 6, 16, 32, 16, 2, 1, 1, 1 }},
        // W28C480K192
        {GFX900, 1, 1, 28, 28, 480, 192, 1, false, "Conv1x1.cl", { 4, 16, 64, 32, 4, 2, 1, 1 }},
        // W14C864K224
        {GFX900, 1, 1, 14, 14, 864, 224, 1, false, "Conv1x1.cl", { 6, 16, 32, 32, 2, 2, 1, 1 }},
        // W14C864K64
        {GFX900, 1, 1, 14, 14, 864, 64, 1, false, "Conv1x1.cl", { 9, 16, 16, 16, 1, 1, 1, 1 }},
        // W14CXK128
        {GFX900, 1, 1, 14, 14, 0, 128, 1, false, "Conv1x1.cl", { 9, 16, 16, 32, 2, 1, 1, 1 }},
        // W14CXK192
        {GFX900, 1, 1, 14, 14, 0, 192, 1, false, "Conv1x1.cl", { 9, 16, 16, 32, 2, 1, 1, 1 }},
        // W14CXK160
        {GFX900, 1, 1, 14, 14, 0, 160, 1, false, "Conv1x1.cl", { 9, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK352
        {GFX900, 1, 1, 7, 7, 0, 352, 1, false, "Conv1x1.cl", { 8, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK192
        {GFX900, 1, 1, 7, 7, 0, 192, 1, false, "Conv1x1.cl", { 12, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK128
        {GFX900, 1, 1, 7, 7, 0, 128, 1, false, "Conv1x1.cl", { 12, 16, 16, 32, 2, 1, 1, 1 }},

        // VGG-SSD
        // W19C1024K1024
        {GFX900, 1, 1, 19, 19, 1024, 1024, 1, false, "Conv1x1.cl", { 4, 16, 64, 64, 8, 2, 1, 1 }},
        // W19C1024K256
        {GFX900, 1, 1, 19, 19, 1024, 256, 1, false, "Conv1x1.cl", { 4, 16, 64, 32, 4, 2, 1, 1 }},
        // W10C512K128
        {GFX900, 1, 1, 10, 10, 512, 128, 1, false, "Conv1x1.cl", { 8, 16, 32, 16, 2, 1, 1, 1 }},
        // W5C256K128
        {GFX900, 1, 1, 5, 5, 256, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},
        // W3C256K128
        {GFX900, 1, 1, 3, 3, 256, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},

        // Mobilenet-SSD
        // W150C32K64
        {GFX900, 1, 1, 150, 150, 32, 64, 1, false, "Conv1x1.cl", { 1, 16, 64, 64, 8, 2, 1, 1 }},
        // W75CXK128
        {GFX900, 1, 1, 75, 75, 0, 128, 1, false, "Conv1x1.cl", { 1, 16, 64, 64, 8, 2, 1, 1 }},
        // W38CXK256
        {GFX900, 1, 1, 38, 38, 0, 256, 1, false, "Conv1x1.cl", { 2, 16, 64, 64, 8, 2, 1, 1 }},
        // W19CXK512
        {GFX900, 1, 1, 19, 19, 0, 512, 1, false, "Conv1x1.cl", { 4, 16, 64, 64, 8, 2, 1, 1 }},
        // W10CXK1024
        {GFX900, 1, 1, 10, 10, 0, 1024, 1, false, "Conv1x1.cl", { 8, 16, 64, 64, 8, 2, 1, 1 }},
        // W10C1024K256
        {GFX900, 1, 1, 10, 10, 1024, 256, 1, false, "Conv1x1.cl", { 8, 16, 32, 32, 2, 2, 1, 1 }},
        // W5C512K128
        {GFX900, 1, 1, 5, 5, 512, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},
        // W2C256K64
        {GFX900, 1, 1, 2, 2, 256, 64, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},

        // Inception
        // W56C96K128
        {GFX900, 2, 1, 56, 96, 128, 1, false, "Conv1x1.cl", { 2, 16, 64, 32, 4, 2, 1, 1 }},
        // W28CXK96
        {GFX900, 2, 1, 28, 28, 0, 96, 1, false, "Conv1x1.cl", { 6, 16, 16, 32, 2, 1, 1, 1 }},
        // W28C288K48
        {GFX900, 2, 1, 28, 28, 288, 48, 1, false, "Conv1x1.cl", { 6, 16, 32, 16, 2, 1, 1, 1 }},
        // W28C480K192
        {GFX900, 2, 1, 28, 28, 480, 192, 1, false, "Conv1x1.cl", { 4, 16, 64, 32, 4, 2, 1, 1 }},
        // W14C864K224
        {GFX900, 2, 1, 14, 14, 864, 224, 1, false, "Conv1x1.cl", { 6, 16, 32, 32, 2, 2, 1, 1 }},
        // W14C864K64
        {GFX900, 2, 1, 14, 14, 864, 64, 1, false, "Conv1x1.cl", { 9, 16, 16, 16, 1, 1, 1, 1 }},
        // W14CXK128
        {GFX900, 2, 1, 14, 14, 0, 128, 1, false, "Conv1x1.cl", { 9, 16, 16, 32, 2, 1, 1, 1 }},
        // W14CXK192
        {GFX900, 2, 1, 14, 14, 0, 192, 1, false, "Conv1x1.cl", { 9, 16, 16, 32, 2, 1, 1, 1 }},
        // W14CXK160
        {GFX900, 2, 1, 14, 14, 0, 160, 1, false, "Conv1x1.cl", { 9, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK352
        {GFX900, 2, 1, 7, 7, 0, 352, 1, false, "Conv1x1.cl", { 8, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK192
        {GFX900, 2, 1, 7, 7, 0, 192, 1, false, "Conv1x1.cl", { 12, 16, 16, 32, 2, 1, 1, 1 }},
        // W7CXK128
        {GFX900, 2, 1, 7, 7, 0, 128, 1, false, "Conv1x1.cl", { 12, 16, 16, 32, 2, 1, 1, 1 }},

        // VGG-SSD
        // W19C1024K1024
        {GFX900, 2, 1, 19, 19, 1024, 1024, 1, false, "Conv1x1.cl", { 4, 16, 64, 64, 8, 2, 1, 1 }},
        // W19C1024K256
        {GFX900, 2, 1, 19, 19, 1024, 256, 1, false, "Conv1x1.cl", { 4, 16, 64, 32, 4, 2, 1, 1 }},
        // W10C512K128
        {GFX900, 2, 1, 10, 10, 512, 128, 1, false, "Conv1x1.cl", { 8, 16, 32, 16, 2, 1, 1, 1 }},
        // W5C256K128
        {GFX900, 2, 1, 5, 5, 256, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},
        // W3C256K128
        {GFX900, 2, 1, 3, 3, 256, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},

        // Mobilenet-SSD
        // W150C32K64
        {GFX900, 2, 1, 150, 150, 32, 64, 1, false, "Conv1x1.cl", { 1, 16, 64, 64, 8, 2, 1, 1 }},
        // W75CXK128
        {GFX900, 2, 1, 75, 75, 0, 128, 1, false, "Conv1x1.cl", { 1, 16, 64, 64, 8, 2, 1, 1 }},
        // W38CXK256
        {GFX900, 2, 1, 38, 38, 0, 256, 1, false, "Conv1x1.cl", { 2, 16, 64, 64, 8, 2, 1, 1 }},
        // W19CXK512
        {GFX900, 2, 1, 19, 19, 0, 512, 1, false, "Conv1x1.cl", { 4, 16, 64, 64, 8, 2, 1, 1 }},
        // W10CXK1024
        {GFX900, 2, 1, 10, 10, 0, 1024, 1, false, "Conv1x1.cl", { 8, 16, 64, 64, 8, 2, 1, 1 }},
        // W10C1024K256
        {GFX900, 2, 1, 10, 10, 1024, 256, 1, false, "Conv1x1.cl", { 8, 16, 32, 32, 2, 2, 1, 1 }},
        // W5C512K128
        {GFX900, 2, 1, 5, 5, 512, 128, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},
        // W2C256K64
        {GFX900, 2, 1, 2, 2, 256, 64, 1, false, "Conv1x1.cl", { 8, 16, 16, 16, 1, 1, 1, 1 }},


        // Wildcat
        {GFX900, 1, 0, 0, 0, 0, 0, 4, false, "Conv1x1.cl", { 0, 0, 0, 0, 0, 0, 0, 0 }},
        // Wildcat
        {GFX900, 2, 0, 0, 0, 0, 0, 4, false, "Conv1x1.cl", { 0, 0, 0, 0, 0, 0, 0, 0 }},

    };
};

} // namespace miopen

#endif
