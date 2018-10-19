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
#include "miopen/solver_conv_common.hpp"

namespace miopen {
const int WidthArray[6] = {1, 7, 14, 28, 56, 112};
const int ChannelArray[18] =
        {16, 24, 32, 64, 96, 128, 144, 160, 192, 256, 320, 384, 512, 576, 960, 1024, 1280, 2048};
const int OutputNumArray[19] = {16,
                                24,
                                32,
                                64,
                                96,
                                128,
                                144,
                                160,
                                192,
                                256,
                                320,
                                384,
                                512,
                                576,
                                960,
                                1000,
                                1024,
                                1280,
                                2048};
const int BatchArray[5]      = {1, 2, 4, 8, 32};

void ConvCommon::init() {

    if (conv1x1type.size() > 0)
        return;

    Conv1x1Type* tempType;

    //*********************************************************
    // GFX803
    //*********************************************************
    //---------------------------------------------
    // stride = 1
    //---------------------------------------------
    // width = 1
    // Mobilenet, W1C1024K1000
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 1024;
    tempType->width       = 1;
    tempType->output_num  = 1000;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1FC7.cl";
    conv1x1type.push_back(tempType);

    // MobilenetV2, W1C1280K1000
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 1280;
    tempType->width       = 1;
    tempType->output_num  = 1000;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1FC7.cl";
    conv1x1type.push_back(tempType);

    // width = 7
    // MobilenetV2, C160H7W7K960S1 but not implement

    // MobilenetV2, W7C320K1280POOL but not implement

    // Mobilenet, C512H7W7K1024S1 but not implement

    // Resnet, W7C512K2048
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 512;
    tempType->width       = 7;
    tempType->output_num  = 2048;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {4, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W7C512K2048
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 512;
    tempType->width       = 7;
    tempType->output_num  = 2048;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {4, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // MobilenetV2, W7CXK160
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 576;
    tempType->width       = 7;
    tempType->output_num  = 160;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1CXH7W7K160.cl";
    conv1x1type.push_back(tempType);

    // MobilenetV2, W7CXK160
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 960;
    tempType->width       = 7;
    tempType->output_num  = 160;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1CXH7W7K160.cl";
    conv1x1type.push_back(tempType);

    // MobilenetV2, W7CXK320
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 960;
    tempType->width       = 7;
    tempType->output_num  = 320;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1CXH7W7K320.cl";
    conv1x1type.push_back(tempType);

    // Mobilenet, W7C1024K1024POOL but not implement

    // Resnet, W7C2048K512
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 2048;
    tempType->width       = 7;
    tempType->output_num  = 512;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W7C2048K512
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 2048;
    tempType->width       = 7;
    tempType->output_num  = 512;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // width = 14
    // MobilenetV2, C96H14W14K576S1 but not implement

    // Mobilenet, C256H14W14K512S1 but not implement

    // Resnet, W14C256K1024
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 256;
    tempType->width       = 14;
    tempType->output_num  = 1024;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {2, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // MobilenetV2, W14CXK96
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 384;
    tempType->width       = 14;
    tempType->output_num  = 96;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1CXH14W14K96.cl";
    conv1x1type.push_back(tempType);

    // Mobilenet, C512H14W14K512S1 but not implement

    // MobilenetV2, W14CXK96
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 576;
    tempType->width       = 14;
    tempType->output_num  = 96;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1CXH14W14K96.cl";
    conv1x1type.push_back(tempType);

    // Resnet, W14C1024K256
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 256;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W14C1024K256
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 256;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // YOLO, C1024H14W14K512S1 but not implement

    // width = 28
    // MobilenetV2, C32H28W28K192S1 but not implement

    // MobilenetV2, C64H28W28K384S1 but not implement

    // Mobilenet, C128H28W28K256S1 but not implement

    // Resnet, C128H28W28K512S1 but not implement

    // MobilenetV2, C144H28W28K32S1 but not implement

    // MobilenetV2, C192H28W28K32S1 but not implement

    // MobilenetV2, C192H28W28K64S1 but not implement

    // Mobilenet, C256H28W28K256S1 but not implement

    // MobilenetV2, C384H28W28K64S1 but not implement

    // Resnet, W28C512K128
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 512;
    tempType->width       = 28;
    tempType->output_num  = 128;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {4, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // YOLO, C512H28W28K256S1 but not implement

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
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 256;
    tempType->width       = 56;
    tempType->output_num  = 64;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {2, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

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
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 512;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W14C1024K512S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 512;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W14C1024K2048S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 2048;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W14C1024K2048S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 2048;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // width = 28
    // Resnet, W28C512K256S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 512;
    tempType->width       = 28;
    tempType->output_num  = 256;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W28C512K256S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 512;
    tempType->width       = 28;
    tempType->output_num  = 256;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W28C512K1024S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 512;
    tempType->width       = 28;
    tempType->output_num  = 1024;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {2, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W28C512K1024S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 512;
    tempType->width       = 28;
    tempType->output_num  = 1024;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {2, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // width = 56
    // Resnet, W56C256K128S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 256;
    tempType->width       = 56;
    tempType->output_num  = 128;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {4, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W56C256K128S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 256;
    tempType->width       = 56;
    tempType->output_num  = 128;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {4, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // Resnet, C256H56W56K512S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 256;
    tempType->width       = 56;
    tempType->output_num  = 512;
    tempType->batch       = 1;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1C256H56W56K512S2.cl";
    conv1x1type.push_back(tempType);

    // Resnet, C256H56W56K512S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 256;
    tempType->width       = 56;
    tempType->output_num  = 512;
    tempType->batch       = 2;
    tempType->dev         = GFX803;
    tempType->kernel_name = "Conv1x1C256H56W56K512S2.cl";
    conv1x1type.push_back(tempType);

    //*********************************************************
    // GFX900
    //*********************************************************
    //---------------------------------------------
    // stride = 1
    //---------------------------------------------
    // width = 1
    // Mobilenet, W1C1024K1000
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 1024;
    tempType->width       = 1;
    tempType->output_num  = 1000;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1FC7.cl";
    conv1x1type.push_back(tempType);

    // MobilenetV2, W1C1280K1000
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 1280;
    tempType->width       = 1;
    tempType->output_num  = 1000;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1FC7.cl";
    conv1x1type.push_back(tempType);

    // width = 7
    // MobilenetV2, C160H7W7K960S1 but not implement

    // MobilenetV2, W7C320K1280POOL but not implement

    // Mobilenet, C512H7W7K1024S1 but not implement

    // Resnet, W7C512K2048
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 512;
    tempType->width       = 7;
    tempType->output_num  = 2048;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {4, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W7C512K2048
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 512;
    tempType->width       = 7;
    tempType->output_num  = 2048;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {4, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // MobilenetV2, W7CXK160
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 576;
    tempType->width       = 7;
    tempType->output_num  = 160;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1CXH7W7K160.cl";
    conv1x1type.push_back(tempType);

    // MobilenetV2, W7CXK160
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 960;
    tempType->width       = 7;
    tempType->output_num  = 160;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1CXH7W7K160.cl";
    conv1x1type.push_back(tempType);

    // MobilenetV2, W7CXK320
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 960;
    tempType->width       = 7;
    tempType->output_num  = 320;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1CXH7W7K320.cl";
    conv1x1type.push_back(tempType);

    // Mobilenet, W7C1024K1024POOL but not implement

    // Resnet, W7C2048K512
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 2048;
    tempType->width       = 7;
    tempType->output_num  = 512;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W7C2048K512
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 2048;
    tempType->width       = 7;
    tempType->output_num  = 512;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // width = 14
    // MobilenetV2, C96H14W14K576S1 but not implement

    // Mobilenet, C256H14W14K512S1 but not implement

    // Resnet, W14C256K1024
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 256;
    tempType->width       = 14;
    tempType->output_num  = 1024;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {2, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // MobilenetV2, W14CXK96
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 384;
    tempType->width       = 14;
    tempType->output_num  = 96;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1CXH14W14K96.cl";
    conv1x1type.push_back(tempType);

    // Mobilenet, C512H14W14K512S1 but not implement

    // MobilenetV2, W14CXK96
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 576;
    tempType->width       = 14;
    tempType->output_num  = 96;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1CXH14W14K96.cl";
    conv1x1type.push_back(tempType);

    // Resnet, W14C1024K256
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 256;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W14C1024K256
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 256;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // YOLO, C1024H14W14K512S1 but not implement

    // width = 28
    // MobilenetV2, C32H28W28K192S1 but not implement

    // MobilenetV2, C64H28W28K384S1 but not implement

    // Mobilenet, C128H28W28K256S1 but not implement

    // Resnet, C128H28W28K512S1 but not implement

    // MobilenetV2, C144H28W28K32S1 but not implement

    // MobilenetV2, C192H28W28K32S1 but not implement

    // MobilenetV2, C192H28W28K64S1 but not implement

    // Mobilenet, C256H28W28K256S1 but not implement

    // MobilenetV2, C384H28W28K64S1 but not implement

    // Resnet, W28C512K128
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 512;
    tempType->width       = 28;
    tempType->output_num  = 128;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {4, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // YOLO, C512H28W28K256S1 but not implement

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
    tempType              = new Conv1x1Type();
    tempType->stride      = 1;
    tempType->channel     = 256;
    tempType->width       = 56;
    tempType->output_num  = 64;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {2, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

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
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 512;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W14C1024K512S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 512;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W14C1024K2048S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 2048;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W14C1024K2048S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 1024;
    tempType->width       = 14;
    tempType->output_num  = 2048;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // width = 28
    // Resnet, W28C512K256S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 512;
    tempType->width       = 28;
    tempType->output_num  = 256;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W28C512K256S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 512;
    tempType->width       = 28;
    tempType->output_num  = 256;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {8, 32, 32, 64, 4, 2, 3, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W28C512K1024S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 512;
    tempType->width       = 28;
    tempType->output_num  = 1024;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {2, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W28C512K1024S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 512;
    tempType->width       = 28;
    tempType->output_num  = 1024;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {2, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // width = 56
    // Resnet, W56C256K128S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 256;
    tempType->width       = 56;
    tempType->output_num  = 128;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {4, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, W56C256K128S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 256;
    tempType->width       = 56;
    tempType->output_num  = 128;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1Atomic.cl";
    tempType->params      = {4, 32, 32, 64, 4, 2, 1, 1};
    conv1x1type.push_back(tempType);

    // Resnet, C256H56W56K512S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 256;
    tempType->width       = 56;
    tempType->output_num  = 512;
    tempType->batch       = 1;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1C256H56W56K512S2.cl";
    conv1x1type.push_back(tempType);

    // Resnet, C256H56W56K512S2
    tempType              = new Conv1x1Type();
    tempType->stride      = 2;
    tempType->channel     = 256;
    tempType->width       = 56;
    tempType->output_num  = 512;
    tempType->batch       = 2;
    tempType->dev         = GFX900;
    tempType->kernel_name = "Conv1x1C256H56W56K512S2.cl";
    conv1x1type.push_back(tempType);
}

Conv1x1Type*
ConvCommon::getKernelInfo(int dev, int stride, int channel, int width, int output_num) {

    init();

    Conv1x1Type* mType;
    for (int i = 0; i < conv1x1type.size(); i++) {
        if (conv1x1type[i]->dev == dev && conv1x1type[i]->stride == stride
            && conv1x1type[i]->channel == channel && conv1x1type[i]->width == width
            && conv1x1type[i]->output_num == output_num) {
            mType = conv1x1type[i];
            ALOGD("Got kernel:" << mType->kernel_name << "!!");
            return mType;
        }
    }
    return NULL;
}

} // namespace miopen
