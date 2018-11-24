/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
    int width;
    int channel;
    int output_num;
    std::string kernel_name;
    Conv11Param params;
};

class ConvCommon {
public:
    ConvCommon() {}
    ~ConvCommon() {}

    bool getKernelInfo(int dev, int batch, int stride, int channel, int width, int output_num,
                       Conv1x1Type& mType);

private:
    std::vector<Conv1x1Type> conv1x1type {

        //*********************************************************
        // GFX803
        //*********************************************************
        //---------------------------------------------
        // stride = 1
        //---------------------------------------------

        // tensile for Resnet50, Resnet101
        //{GFX803, 1, 1, 7, 512, 2048, "ConvFwd1x1_7x7x512x2048x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        //{GFX803, 1, 1, 7, 2048, 512, "ConvFwd1x1_7x7x2048x512x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 14, 256, 1024, "ConvFwd1x1_14x14x256x1024x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 14, 1024, 256, "ConvFwd1x1_14x14x1024x256x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 28, 128, 512, "ConvFwd1x1_28x28x128x512x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 28, 512, 128, "ConvFwd1x1_28x28x512x128x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 56, 64, 64, "ConvFwd1x1_56x56x64x64x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 56, 64, 256, "ConvFwd1x1_56x56x64x256x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 1, 1, 56, 256, 64, "ConvFwd1x1_56x56x256x64x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        //{GFX803, 2, 1, 7, 512, 2048, "ConvFwd1x1_7x7x512x2048x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        //{GFX803, 2, 1, 7, 2048, 512, "ConvFwd1x1_7x7x2048x512x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 14, 256, 1024, "ConvFwd1x1_14x14x256x1024x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 14, 1024, 256, "ConvFwd1x1_14x14x1024x256x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 28, 128, 512, "ConvFwd1x1_28x28x128x512x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 28, 512, 128, "ConvFwd1x1_28x28x512x128x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 56, 64, 64, "ConvFwd1x1_56x56x64x64x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 56, 64, 256, "ConvFwd1x1_56x56x64x256x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 2, 1, 56, 256, 64, "ConvFwd1x1_56x56x256x64x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 7, 512, 2048, "ConvFwd1x1_7x7x512x2048x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 7, 2048, 512, "ConvFwd1x1_7x7x2048x512x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 14, 256, 1024, "ConvFwd1x1_14x14x256x1024x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 14, 1024, 256, "ConvFwd1x1_14x14x1024x256x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 28, 128, 512, "ConvFwd1x1_28x28x128x512x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 28, 512, 128, "ConvFwd1x1_28x28x512x128x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 56, 64, 64, "ConvFwd1x1_56x56x64x64x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 56, 64, 256, "ConvFwd1x1_56x56x64x256x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX803, 4, 1, 56, 256, 64, "ConvFwd1x1_56x56x256x64x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},


        // width = 1
        // Mobilenet, W1C1024K1000 *max 8

        {GFX803, 8, 1, 1, 1024, 1000, "Conv1x1FC7.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // MobilenetV2, W1C1280K1000 * max 8
        {GFX803, 8, 1, 1, 1280, 1000, "Conv1x1FC7.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // width = 7
        // MobilenetV2, W7C160K960
        {GFX803, 1, 1, 7, 160, 960, "Conv1x1.cl", {2, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7C160K960 *2
        {GFX803, 2, 1, 7, 160, 960, "Conv1x1.cl", {2, 16, 16, 16, 1, 1, 3, 1}},
        // MobilenetV2, W7C320K1280POOL
        {GFX803, 1, 1, 7, 320, 1280, "Conv1x1C320H7W7K1280Pool.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // MobilenetV2, W7C320K1280POOL *2
        {GFX803, 1, 1, 7, 320, 1280, "Conv1x1C320H7W7K1280Pool.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Mobilenet, W7C512K1024
        {GFX803, 2, 1, 7, 512, 1024, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 3, 1}},
        // Resnet, W7C512K2048
        {GFX803, 1, 1, 7, 512, 2048, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W7C512K2048 *2
        {GFX803, 2, 1, 7, 512, 2048, "Conv1x1.cl", {2, 32, 32, 32, 2, 2, 3, 1}},
        // MobilenetV2, W7CXK160
        {GFX803, 1, 1, 7, 576, 160, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK160 *2
        {GFX803, 2, 1, 7, 576, 160, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 3, 1}},
        // MobilenetV2, W7CXK160
        {GFX803, 1, 1, 7, 960, 160, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK160 *2
        {GFX803, 2, 1, 7, 960, 160, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 3, 1}},
        // MobilenetV2, W7CXK320
        {GFX803, 1, 1, 7, 960, 320, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK320 *2
        {GFX803, 2, 1, 7, 960, 320, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 3, 1}},
        // Mobilenet, W7C1024K1024POOL
        {GFX803, 1, 1, 7, 1024, 1024, "Conv1x1C1024H7W7K1024Pool.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Mobilenet, W7C1024K1024POOL *2
        {GFX803, 2, 1, 7, 1024, 1024, "Conv1x1C1024H7W7K1024Pool.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Resnet, W7C2048K512
        {GFX803, 1, 1, 7, 2048, 512, "Conv1x1.cl", {8, 32, 32, 32, 2, 2, 1, 1}},
        // Resnet, W7C2048K512 *2
        {GFX803, 2, 1, 7, 2048, 512, "Conv1x1.cl", {8, 32, 32, 32, 2, 2, 3, 1}},
        // width = 14
        // MobilenetV2, W14C96K576
        {GFX803, 1, 1, 14, 96, 576, "Conv1x1.cl", {2, 16, 32, 32, 4, 1, 1, 1}},
        // MobilenetV2, W14C96K576 *2
        {GFX803, 2, 1, 14, 96, 576, "Conv1x1.cl", {2, 16, 32, 32, 4, 1, 3, 1}},
        // Mobilenet, W14C256K512
        {GFX803, 2, 1, 14, 256, 512, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 3, 1}},
        // MobilenetV2, W14CXK96
        {GFX803, 1, 1, 14, 384, 96, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // MobilenetV2, W14CXK96 *2
        {GFX803, 2, 1, 14, 384, 96, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 3, 1}},
        // Mobilenet, W14C512K512
        {GFX803, 2, 1, 14, 512, 512, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 3, 1}},
        // MobilenetV2, W14CXK96
        {GFX803, 1, 1, 14, 576, 96, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // MobilenetV2, W14CXK96 *2
        {GFX803, 2, 1, 14, 576, 96, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 3, 1}},
        // YOLO, W14C1024K512
        {GFX803, 1, 1, 14, 1024, 512, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // YOLO, W14C1024K512 *2
        {GFX803, 2, 1, 14, 1024, 512, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 3, 1}},

        // width = 28
        // MobilenetV2, C32H28W28K192S1 but not implement

        // MobilenetV2, C64H28W28K384S1 but not implement

        // Mobilenet, C128H28W28K256S1 but not implement

        // Resnet, C128H28W28K512S1 but not implement

        // MobilenetV2, W28C144K32
        {GFX803, 2, 1, 28, 144, 32, "Conv1x1.cl", {1, 16, 16, 16, 1, 1, 3, 1}},
        // MobilenetV2, W28C192K32
        {GFX803, 2, 1, 28, 192, 32, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 3, 1}},
        // MobilenetV2, W28C192K64
        {GFX803, 2, 1, 28, 192, 64, "Conv1x1.cl", {4, 16, 16, 64, 2, 2, 3, 1}},
        // Mobilenet, W28C256K256
        {GFX803, 1, 1, 28, 256, 256, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // Mobilenet, W28C256K256 *2
        {GFX803, 2, 1, 28, 256, 256, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // MobilenetV2, W28C384K64
        {GFX803, 2, 1, 28, 384, 64, "Conv1x1.cl", {8, 16, 32, 64, 4, 2, 3, 1}},
        // YOLO, W28C512K256
        {GFX803, 1, 1, 28, 512, 256, "Conv1x1.cl", {2, 32, 32, 64, 4, 2, 1, 1}},
        // YOLO, W28C512K256 *2
        {GFX803, 2, 1, 28, 512, 256, "Conv1x1.cl", {2, 32, 32, 64, 4, 2, 3, 1}},
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
        {GFX803, 1, 2, 14, 1024, 512, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C1024K512S2 *2
        {GFX803, 2, 2, 14, 1024, 512, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 3, 1}},
        // Resnet, W14C1024K2048S2
        {GFX803, 1, 2, 14, 1024, 2048, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C1024K2048S2
        {GFX803, 2, 2, 14, 1024, 2048, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 3, 1}},
        // width = 28
        // Resnet, W28C512K256S2
        {GFX803, 1, 2, 28, 512, 256, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W28C512K256S2 *2
        {GFX803, 2, 2, 28, 512, 256, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 3, 1}},
        // Resnet, W28C512K1024S2
        {GFX803, 1, 2, 28, 512, 1024, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // Resnet, W28C512K1024S2 *2
        {GFX803, 2, 2, 28, 512, 1024, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 3, 1}},
        // width = 56
        // Resnet, W56C256K128S2
        {GFX803, 1, 2, 56, 256, 128, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W56C256K128S2 *2
        {GFX803, 2, 2, 56, 256, 128, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 3, 1}},
        // Resnet, W56C256K512S2
        {GFX803, 1, 2, 56, 256, 512, "Conv1x1.cl", {1, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W56C256K512S2 *2
        {GFX803, 2, 2, 56, 256, 512, "Conv1x1.cl", {1, 32, 32, 64, 4, 2, 3, 1}},

        //*********************************************************
        // GFX900
        //*********************************************************
        //---------------------------------------------
        // stride = 1
        //---------------------------------------------

        // tensile for Resnet50, Resnet101
        //{GFX900, 1, 1, 7, 512, 2048, "ConvFwd1x1_7x7x512x2048x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        //{GFX900, 1, 1, 7, 2048, 512, "ConvFwd1x1_7x7x2048x512x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 1, 1, 14, 256, 1024, "ConvFwd1x1_14x14x256x1024x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 1, 1, 14, 1024, 256, "ConvFwd1x1_14x14x1024x256x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 1, 1, 28, 128, 512, "ConvFwd1x1_28x28x128x512x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 1, 1, 28, 512, 128, "ConvFwd1x1_28x28x512x128x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 1, 1, 56, 64, 64, "ConvFwd1x1_56x56x64x64x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 1, 1, 56, 64, 256, "ConvFwd1x1_56x56x64x256x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 1, 1, 56, 256, 64, "ConvFwd1x1_56x56x256x64x1.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        //{GFX900, 2, 1, 7, 512, 2048, "ConvFwd1x1_7x7x512x2048x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        //{GFX900, 2, 1, 7, 2048, 512, "ConvFwd1x1_7x7x2048x512x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 2, 1, 14, 256, 1024, "ConvFwd1x1_14x14x256x1024x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 2, 1, 14, 1024, 256, "ConvFwd1x1_14x14x1024x256x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 2, 1, 28, 128, 512, "ConvFwd1x1_28x28x128x512x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 2, 1, 28, 512, 128, "ConvFwd1x1_28x28x512x128x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 2, 1, 56, 64, 64, "ConvFwd1x1_56x56x64x64x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 2, 1, 56, 64, 256, "ConvFwd1x1_56x56x64x256x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 2, 1, 56, 256, 64, "ConvFwd1x1_56x56x256x64x2.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 4, 1, 7, 512, 2048, "ConvFwd1x1_7x7x512x2048x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 4, 1, 7, 2048, 512, "ConvFwd1x1_7x7x2048x512x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 4, 1, 14, 256, 1024, "ConvFwd1x1_14x14x256x1024x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 4, 1, 14, 1024, 256, "ConvFwd1x1_14x14x1024x256x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 4, 1, 28, 128, 512, "ConvFwd1x1_28x28x128x512x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 4, 1, 28, 512, 128, "ConvFwd1x1_28x28x512x128x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 4, 1, 56, 64, 64, "ConvFwd1x1_56x56x64x64x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 4, 1, 56, 64, 256, "ConvFwd1x1_56x56x64x256x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},
        {GFX900, 4, 1, 56, 256, 64, "ConvFwd1x1_56x56x256x64x4.s", {0, 0, 0, 0, 0, 0, 0, 0}},


        // width = 1
        // Mobilenet, W1C1024K1000 *max 8
        {GFX900, 8, 1, 1, 1024, 1000, "Conv1x1FC7.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // MobilenetV2, W1C1280K1000 *max 8
        {GFX900, 8, 1, 1, 1280, 1000, "Conv1x1FC7.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // width = 7
        // MobilenetV2, W7C160K960
        {GFX900, 1, 1, 7, 160, 960, "Conv1x1.cl", {2, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7C160K960 *2
        {GFX900, 2, 1, 7, 160, 960, "Conv1x1.cl", {2, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7C320K1280POOL
        {GFX900, 1, 1, 7, 320, 1280, "Conv1x1C320H7W7K1280Pool.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // MobilenetV2, W7C320K1280POOL *2
        {GFX900, 2, 1, 7, 320, 1280, "Conv1x1C320H7W7K1280Pool.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Mobilenet, W7C512K1024
        {GFX900, 2, 1, 7, 512, 1024, "Conv1x1.cl", {4, 32, 32, 32, 4, 1, 1, 1}},
        // Resnet, W7C512K2048
        {GFX900, 1, 1, 7, 512, 2048, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W7C512K2048 *2
        {GFX900, 2, 1, 7, 512, 2048, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // MobilenetV2, W7CXK160
        {GFX900, 1, 1, 7, 576, 160, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK160 *2
        {GFX900, 2, 1, 7, 576, 160, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK160
        {GFX900, 1, 1, 7, 960, 160, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK160 *2
        {GFX900, 2, 1, 7, 960, 160, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK320
        {GFX900, 1, 1, 7, 960, 320, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W7CXK320 *2
        {GFX900, 2, 1, 7, 960, 320, "Conv1x1.cl", {4, 16, 16, 16, 1, 1, 1, 1}},
        // Mobilenet, W7C1024K1024POOL
        {GFX900, 1, 1, 7, 1024, 1024, "Conv1x1C1024H7W7K1024Pool.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Mobilenet, W7C1024K1024POOL *2
        {GFX900, 2, 1, 7, 1024, 1024, "Conv1x1C1024H7W7K1024Pool.cl", {0, 0, 0, 0, 0, 0, 0, 0}},
        // Resnet, W7C2048K512
        {GFX900, 1, 1, 7, 2048, 512, "Conv1x1.cl", {8, 32, 32, 32, 4, 1, 1, 1}},
        // Resnet, W7C2048K512 *2
        {GFX900, 2, 1, 7, 2048, 512, "Conv1x1.cl", {8, 32, 32, 32, 4, 1, 1, 1}},
        // width = 14
        // MobilenetV2, W14C96K576
        {GFX900, 1, 1, 14, 96, 576, "Conv1x1.cl", {2, 16, 32, 32, 4, 1, 1, 1}},
        // MobilenetV2, W14C96K576 *2
        {GFX900, 2, 1, 14, 96, 576, "Conv1x1.cl", {2, 16, 32, 32, 4, 1, 1, 1}},
        // Mobilenet, W14C256K512
        {GFX900, 2, 1, 14, 256, 512, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // MobilenetV2, W14CXK96
        {GFX900, 1, 1, 14, 384, 96, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // MobilenetV2, W14CXK96 *2
        {GFX900, 2, 1, 14, 384, 96, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // Mobilenet, W14C512K512
        {GFX900, 2, 1, 14, 512, 512, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // MobilenetV2, W14CXK96
        {GFX900, 1, 1, 14, 576, 96, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // MobilenetV2, W14CXK96 *2
        {GFX900, 2, 1, 14, 576, 96, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // YOLO, W14C1024K512
        {GFX900, 1, 1, 14, 1024, 512, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // YOLO,  W14C1024K512 *2
        {GFX900, 2, 1, 14, 1024, 512, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},

        // width = 28
        // MobilenetV2, C32H28W28K192S1 but not implement

        // MobilenetV2, C64H28W28K384S1 but not implement

        // Mobilenet, C128H28W28K256S1 but not implement

        // Resnet, C128H28W28K512S1 but not implement

        // MobilenetV2, W28C144K32
        {GFX900, 2, 1, 28, 144, 32, "Conv1x1.cl", {1, 16, 16, 16, 1, 1, 1, 1}},
        // MobilenetV2, W28C192K32
        {GFX900, 2, 1, 28, 192, 32, "Conv1x1.cl", {4, 16, 16, 32, 2, 1, 1, 1}},
        // MobilenetV2, W28C192K64
        {GFX900, 2, 1, 28, 192, 64, "Conv1x1.cl", {4, 16, 16, 64, 4, 1, 1, 1}},
        // Mobilenet, W28C256K256
        {GFX900, 1, 1, 28, 256, 256, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // Mobilenet, W28C256K256 *2
        {GFX900, 2, 1, 28, 256, 256, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // MobilenetV2, W28C384K64
        {GFX900, 2, 1, 28, 384, 64, "Conv1x1.cl", {4, 16, 32, 32, 4, 1, 1, 1}},
        // YOLO, W28C512K256
        {GFX900, 1, 1, 28, 512, 256, "Conv1x1.cl", {2, 32, 32, 64, 4, 2, 1, 1}},
        // YOLO, W28C512K256 *2
        {GFX900, 2, 1, 28, 512, 256, "Conv1x1.cl", {2, 32, 32, 64, 4, 2, 1, 1}},
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
        {GFX900, 1, 2, 14, 1024, 512, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C1024K512S2 *2
        {GFX900, 2, 2, 14, 1024, 512, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C1024K2048S2
        {GFX900, 1, 2, 14, 1024, 2048, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W14C1024K2048S2 *2
        {GFX900, 2, 2, 14, 1024, 2048, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // width = 28
        // Resnet, W28C512K256S2
        {GFX900, 1, 2, 28, 512, 256, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W28C512K256S2 *2
        {GFX900, 2, 2, 28, 512, 256, "Conv1x1.cl", {8, 32, 32, 64, 4, 2, 3, 1}},
        // Resnet, W28C512K1024S2
        {GFX900, 1, 2, 28, 512, 1024, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // Resnet, W28C512K1024S2 *2
        {GFX900, 2, 2, 28, 512, 1024, "Conv1x1.cl", {1, 32, 32, 32, 2, 2, 1, 1}},
        // width = 56
        // Resnet, W56C256K128S2
        {GFX900, 1, 2, 56, 256, 128, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W56C256K128S2 *2
        {GFX900, 2, 2, 56, 256, 128, "Conv1x1.cl", {4, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W56C256K512S2
        {GFX900, 1, 2, 56, 256, 512, "Conv1x1.cl", {1, 32, 32, 64, 4, 2, 1, 1}},
        // Resnet, W56C256K512S2 *2
        {GFX900, 2, 2, 56, 256, 512, "Conv1x1.cl", {1, 32, 32, 64, 4, 2, 1, 1}}
    };
};

} // namespace miopen

#endif
