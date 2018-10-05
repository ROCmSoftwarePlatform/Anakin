/*******************************************************************************
 *solver_conv_common.cpp
 *Created on: Oct. 5, 2018
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

    Conv1x1Type tempType;

    for (int dev_type = 0; dev_type < 2; dev_type++) {
        for (int tmp_stride = 1; tmp_stride <= 2; tmp_stride++) {
            for (int i = 0; i < sizeof(WidthArray) / sizeof(int); i++) {
                for (int j = 0; j < sizeof(ChannelArray) / sizeof(int); j++) {
                    for (int k = 0; k < sizeof(OutputNumArray) / sizeof(int); k++) {
                        for (int n = 0; n < sizeof(BatchArray) / sizeof(int); n++) {
                            tempType.stride     = tmp_stride;
                            tempType.channel    = ChannelArray[j];
                            tempType.width      = WidthArray[i];
                            tempType.output_num = OutputNumArray[k];
                            tempType.batch      = BatchArray[n];

                            if (dev_type == 0) {
                                tempType.dev = GFX803;
                                if (tmp_stride == 1) {
                                    if (tempType.width > 56)
                                        continue; // workaround to speed up first
                                    if (tempType.width == 1) {
                                        if ((tempType.channel != 1024 || tempType.channel != 1280)
                                            && tempType.output_num != 1000)
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 1024) {
                                            if (tempType.output_num == 1000) {
                                                if (tempType.batch == 1) { // Mobilenet,
                                                                           // W1C1024K1000
                                                    tempType.kernel_name = "Conv1x1FC7.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 1280) {
                                            if (tempType.output_num == 1000) {
                                                if (tempType.batch
                                                    == 1) { // MobilenetV2, W1C1280K1000
                                                    tempType.kernel_name = "Conv1x1FC7.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        }
                                    } else if (tempType.width == 7) {
                                        if ((tempType.channel != 512 && tempType.output_num != 2048)
                                            && (tempType.channel != 576
                                                && tempType.output_num != 160)
                                            && (tempType.channel != 960
                                                && (tempType.output_num != 160
                                                    || tempType.output_num != 320))
                                            && (tempType.channel != 2048
                                                && tempType.output_num != 512))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 160) {
                                            if (tempType.output_num == 960) {
                                                // MobilenetV2, C160H7W7K960S1 but not implement
                                            }
                                        } else if (tempType.channel == 320) {
                                            if (tempType.output_num == 1280) {
                                                // MobilenetV2, W7C320K1280POOL but not implement
                                            }
                                        } else if (tempType.channel == 512) {
                                            if (tempType.output_num == 1024) {
                                                // Mobilenet, C512H7W7K1024S1 but not implement
                                            } else if (tempType.output_num == 2048) {
                                                if (tempType.batch == 1) { // Resnet, W7C512K2048
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {4, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {4, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 576) {
                                            if (tempType.output_num == 160) {
                                                if (tempType.batch == 1) { // MobilenetV2, W7CXK160
                                                    tempType.kernel_name = "Conv1x1CXH7W7K160.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 960) {
                                            if (tempType.output_num == 160) {
                                                if (tempType.batch == 1) { // MobilenetV2, W7CXK160
                                                    tempType.kernel_name = "Conv1x1CXH7W7K160.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 320) {
                                                if (tempType.batch == 1) { // MobilenetV2, W7CXK320
                                                    tempType.kernel_name = "Conv1x1CXH7W7K320.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 1024) {
                                            if (tempType.output_num == 1024) {
                                                // Mobilenet, W7C1024K1024POOL but not implement
                                            }
                                        } else if (tempType.channel == 2048) {
                                            if (tempType.output_num == 512) {
                                                if (tempType.batch == 1) { // Resnet, W7C2048K512
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        }
                                    } else if (tempType.width == 14) {
                                        if ((tempType.channel != 256 && tempType.output_num != 1024)
                                            && (tempType.channel != 384
                                                && tempType.output_num != 96)
                                            && (tempType.channel != 576
                                                && tempType.output_num != 96)
                                            && (tempType.channel != 1024
                                                && tempType.output_num != 256))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 96) {
                                            if (tempType.output_num == 576) {
                                                // MobilenetV2, C96H14W14K576S1 but not implement
                                            } else {
                                                // todo
                                            }
                                        } else if (tempType.channel == 256) {
                                            if (tempType.output_num == 512) {
                                                // Mobilenet, C256H14W14K512S1 but not implement
                                            } else if (tempType.output_num == 1024) {
                                                if (tempType.batch == 2) { // Resnet, W14C256K1024
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {2, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 384) {
                                            if (tempType.output_num == 96) {
                                                if (tempType.batch == 1) { // MobilenetV2, W14CXK96
                                                    tempType.kernel_name = "Conv1x1CXH14W14K96.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 512) {
                                            if (tempType.output_num == 512) {
                                                // Mobilenet, C512H14W14K512S1 but not implement
                                            } else {
                                                // todo
                                            }
                                        } else if (tempType.channel == 576) {
                                            if (tempType.output_num == 96) {
                                                if (tempType.batch == 1) { // MobilenetV2, W14CXK96
                                                    tempType.kernel_name = "Conv1x1CXH14W14K96.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 1024) {
                                            if (tempType.output_num == 256) {
                                                if (tempType.batch == 1) { // Resnet, W14C1024K256
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 512) {
                                                // YOLO, C1024H14W14K512S1 but not implement
                                            }
                                        }
                                    } else if (tempType.width == 28) {
                                        if ((tempType.channel != 512 && tempType.output_num != 128))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 32) {
                                            if (tempType.output_num == 192) {
                                                // MobilenetV2, C32H28W28K192S1 but not implement
                                            }
                                        } else if (tempType.channel == 64) {
                                            if (tempType.output_num == 384) {
                                                // MobilenetV2, C64H28W28K384S1 but not implement
                                            }
                                        } else if (tempType.channel == 128) {
                                            if (tempType.output_num == 256) {
                                                // Mobilenet, C128H28W28K256S1 but not implement
                                            } else if (tempType.output_num == 512) {
                                                // Resnet, C128H28W28K512S1 but not implement
                                            }
                                        } else if (tempType.channel == 144) {
                                            if (tempType.output_num == 32) {
                                                // MobilenetV2, C144H28W28K32S1 but not implement
                                            }
                                        } else if (tempType.channel == 192) {
                                            if (tempType.output_num == 32) {
                                                // MobilenetV2, C192H28W28K32S1 but not implement
                                            } else if (tempType.output_num == 64) {
                                                // MobilenetV2, C192H28W28K64S1 but not implement
                                            }
                                        } else if (tempType.channel == 256) {
                                            if (tempType.output_num == 256) {
                                                // Mobilenet, C256H28W28K256S1 but not implement
                                            }
                                        } else if (tempType.channel == 384) {
                                            if (tempType.output_num == 64) {
                                                // MobilenetV2, C384H28W28K64S1 but not implement
                                            }
                                        } else if (tempType.channel == 512) {
                                            if (tempType.output_num == 128) {
                                                if (tempType.batch == 2) { // Resnet, W28C512K128
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {4, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 256) {
                                                // YOLO, C512H28W28K256S1 but not implement
                                            } else if (tempType.output_num == 512) {
                                                // YOLO, C512H28W28K512S1 but not implement
                                            }
                                        }
                                    } else if (tempType.width == 56) {
                                        if ((tempType.channel != 256 && tempType.output_num != 64))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 24) {
                                            if (tempType.output_num == 144) {
                                                // MobilenetV2, C24H56W56K144S1 but not implement
                                            }
                                        } else if (tempType.channel == 64) {
                                            if (tempType.output_num == 64) {
                                                // Resnet, C64H56W56K64S1 but not implement
                                            } else if (tempType.output_num == 128) {
                                                // Mobilenet, C64H56W56K128S1 but not implement
                                            } else if (tempType.output_num == 256) {
                                                // Resnet, C64H56W56K256S1 but not implement
                                            }
                                        } else if (tempType.channel == 96) {
                                            if (tempType.output_num == 24) {
                                                // MobilenetV2, C96H56W56K24S1 but not implement
                                            }
                                        } else if (tempType.channel == 128) {
                                            if (tempType.output_num == 128) {
                                                // Mobilenet, C128H56W56K128S1 but not implement
                                            }
                                        } else if (tempType.channel == 144) {
                                            if (tempType.output_num == 24) {
                                                // MobilenetV2, C144H56W56K24S1 but not implement
                                            }
                                        } else if (tempType.channel == 192) {
                                            if (tempType.output_num == 128) {
                                                // YOLO, C192H56W56K128S1 but not implement
                                            }
                                        } else if (tempType.channel == 256) {
                                            if (tempType.output_num == 64) {
                                                if (tempType.batch == 2) { // Resnet, W56C256K64
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {2, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 256) {
                                                // YOLO, C256H56W56K256S1 but not implement
                                            }
                                        }
                                    } else { // width = 112
                                        if (tempType.channel == 16) {
                                            if (tempType.output_num == 96) {
                                                // MobilenetV2, C16H112W112K96S1 but not implement
                                            }
                                        } else if (tempType.channel == 32) {
                                            if (tempType.output_num == 16) {
                                                // MobilenetV2, C32H112W112K16S1 but not implement
                                            } else if (tempType.output_num == 32) {
                                                // MobilenetV2, C32H112W112K32S1 but not implement
                                            } else if (tempType.output_num == 64) {
                                                // Mobilenet, C32H112W112K64S1
                                            }
                                        }
                                    }
                                } else { // stride = 2
                                    if (tempType.width < 14)
                                        continue; // workaround to speed up first
                                    if (tempType.width == 14) {
                                        if ((tempType.channel != 1024
                                             && (tempType.output_num != 512
                                                 || tempType.output_num != 2048)))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 1024) {
                                            if (tempType.output_num == 512) {
                                                if (tempType.batch == 1) { // Resnet, W14C1024K512S2
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 2048) {
                                                if (tempType.batch == 1) { // Resnet,
                                                                           // W14C1024K2048S2
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        }
                                    } else if (tempType.width == 28) {
                                        if ((tempType.channel != 512
                                             && (tempType.output_num != 256
                                                 || tempType.output_num != 1024)))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 512) {
                                            if (tempType.output_num == 256) {
                                                if (tempType.batch == 1) { // Resnet, W28C512K256S2
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 1024) {
                                                if (tempType.batch == 1) { // Resnet, W28C512K1024S2
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {2, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {2, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        }
                                    } else if (tempType.width == 56) {
                                        if ((tempType.channel != 256
                                             && (tempType.output_num != 128
                                                 || tempType.output_num != 512)))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 256) {
                                            if (tempType.output_num == 128) {
                                                if (tempType.batch == 1) { // Resnet, W56C256K128S2
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {4, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {4, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 512) {
                                                if (tempType.batch
                                                    <= 2) { // Resnet, C256H56W56K512S2
                                                    tempType.kernel_name =
                                                            "Conv1x1C256H56W56K512S2.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        }
                                    }
                                } // stride = 2
                            } else {
                                tempType.dev = GFX900;
                                if (tmp_stride == 1) {
                                    if (tempType.width > 56)
                                        continue; // workaround to speed up first
                                    if (tempType.width == 1) {
                                        if ((tempType.channel != 1024 || tempType.channel != 1280)
                                            && tempType.output_num != 1000)
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 1024) {
                                            if (tempType.output_num == 1000) {
                                                if (tempType.batch == 1) { // Mobilenet,
                                                                           // W1C1024K1000
                                                    tempType.kernel_name = "Conv1x1FC7.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 1280) {
                                            if (tempType.output_num == 1000) {
                                                if (tempType.batch
                                                    == 1) { // MobilenetV2, W1C1280K1000
                                                    tempType.kernel_name = "Conv1x1FC7.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        }
                                    } else if (tempType.width == 7) {
                                        if ((tempType.channel != 512 && tempType.output_num != 2048)
                                            && (tempType.channel != 576
                                                && tempType.output_num != 160)
                                            && (tempType.channel != 960
                                                && (tempType.output_num != 160
                                                    || tempType.output_num != 320))
                                            && (tempType.channel != 2048
                                                && tempType.output_num != 512))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 160) {
                                            if (tempType.output_num == 960) {
                                                // MobilenetV2, C160H7W7K960S1 but not implement
                                            }
                                        } else if (tempType.channel == 320) {
                                            if (tempType.output_num == 1280) {
                                                // MobilenetV2, W7C320K1280POOL but not implement
                                            }
                                        } else if (tempType.channel == 512) {
                                            if (tempType.output_num == 1024) {
                                                // Mobilenet, C512H7W7K1024S1 but not implement
                                            } else if (tempType.output_num == 2048) {
                                                if (tempType.batch == 1) { // Resnet, W7C512K2048
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {4, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {4, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 576) {
                                            if (tempType.output_num == 160) {
                                                if (tempType.batch == 1) { // MobilenetV2, W7CXK160
                                                    tempType.kernel_name = "Conv1x1CXH7W7K160.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 960) {
                                            if (tempType.output_num == 160) {
                                                if (tempType.batch == 1) { // MobilenetV2, W7CXK160
                                                    tempType.kernel_name = "Conv1x1CXH7W7K160.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 320) {
                                                if (tempType.batch == 1) { // MobilenetV2, W7CXK320
                                                    tempType.kernel_name = "Conv1x1CXH7W7K320.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 1024) {
                                            if (tempType.output_num == 1024) {
                                                // Mobilenet, W7C1024K1024POOL but not implement
                                            }
                                        } else if (tempType.channel == 2048) {
                                            if (tempType.output_num == 512) {
                                                if (tempType.batch == 1) { // Resnet, W7C2048K512
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        }
                                    } else if (tempType.width == 14) {
                                        if ((tempType.channel != 256 && tempType.output_num != 1024)
                                            && (tempType.channel != 384
                                                && tempType.output_num != 96)
                                            && (tempType.channel != 576
                                                && tempType.output_num != 96)
                                            && (tempType.channel != 1024
                                                && tempType.output_num != 256))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 96) {
                                            if (tempType.output_num == 576) {
                                                // MobilenetV2, C96H14W14K576S1 but not implement
                                            } else {
                                                // todo
                                            }
                                        } else if (tempType.channel == 256) {
                                            if (tempType.output_num == 512) {
                                                // Mobilenet, C256H14W14K512S1 but not implement
                                            } else if (tempType.output_num == 1024) {
                                                if (tempType.batch == 2) { // Resnet, W14C256K1024
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {2, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 384) {
                                            if (tempType.output_num == 96) {
                                                if (tempType.batch == 1) { // MobilenetV2, W14CXK96
                                                    tempType.kernel_name = "Conv1x1CXH14W14K96.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 512) {
                                            if (tempType.output_num == 512) {
                                                // Mobilenet, C512H14W14K512S1 but not implement
                                            }
                                        } else if (tempType.channel == 576) {
                                            if (tempType.output_num == 96) {
                                                if (tempType.batch == 1) { // MobilenetV2, W14CXK96
                                                    tempType.kernel_name = "Conv1x1CXH14W14K96.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        } else if (tempType.channel == 1024) {
                                            if (tempType.output_num == 256) {
                                                if (tempType.batch == 1) { // Resnet, W14C1024K256
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 512) {
                                                // YOLO, C1024H14W14K512S1 but not implement
                                            }
                                        }
                                    } else if (tempType.width == 28) {
                                        if ((tempType.channel != 512 && tempType.output_num != 128))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 32) {
                                            if (tempType.output_num == 192) {
                                                // MobilenetV2, C32H28W28K192S1 but not implement
                                            }
                                        } else if (tempType.channel == 64) {
                                            if (tempType.output_num == 384) {
                                                // MobilenetV2, C64H28W28K384S1 but not implement
                                            }
                                        } else if (tempType.channel == 128) {
                                            if (tempType.output_num == 256) {
                                                // Mobilenet, C128H28W28K256S1 but not implement
                                            } else if (tempType.output_num == 512) {
                                                // Resnet, C128H28W28K512S1 but not implement
                                            }
                                        } else if (tempType.channel == 144) {
                                            if (tempType.output_num == 32) {
                                                // MobilenetV2, C144H28W28K32S1 but not implement
                                            }
                                        } else if (tempType.channel == 192) {
                                            if (tempType.output_num == 32) {
                                                // MobilenetV2, C192H28W28K32S1 but not implement
                                            } else if (tempType.output_num == 64) {
                                                // MobilenetV2, C192H28W28K64S1 but not implement
                                            }
                                        } else if (tempType.channel == 256) {
                                            if (tempType.output_num == 256) {
                                                // Mobilenet, C256H28W28K256S1 but not implement
                                            }
                                        } else if (tempType.channel == 384) {
                                            if (tempType.output_num == 64) {
                                                // MobilenetV2, C384H28W28K64S1 but not implement
                                            }
                                        } else if (tempType.channel == 512) {
                                            if (tempType.output_num == 128) {
                                                if (tempType.batch == 2) { // Resnet, W28C512K128
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {4, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 256) {
                                                // YOLO, C512H28W28K256S1 but not implement
                                            } else if (tempType.output_num == 512) {
                                                // YOLO, C512H28W28K512S1 but not implement
                                            }
                                        }
                                    } else if (tempType.width == 56) {
                                        if ((tempType.channel != 256 && tempType.output_num != 64))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 24) {
                                            if (tempType.output_num == 144) {
                                                // MobilenetV2, C24H56W56K144S1 but not implement
                                            }
                                        } else if (tempType.channel == 64) {
                                            if (tempType.output_num == 64) {
                                                // Resnet, C64H56W56K64S1 but not implement
                                            } else if (tempType.output_num == 128) {
                                                // Mobilenet, C64H56W56K128S1 but not implement
                                            } else if (tempType.output_num == 256) {
                                                // Resnet, C64H56W56K256S1 but not implement
                                            }
                                        } else if (tempType.channel == 96) {
                                            if (tempType.output_num == 24) {
                                                // MobilenetV2, C96H56W56K24S1 but not implement
                                            }
                                        } else if (tempType.channel == 128) {
                                            if (tempType.output_num == 128) {
                                                // Mobilenet, C128H56W56K128S1 but not implement
                                            }
                                        } else if (tempType.channel == 144) {
                                            if (tempType.output_num == 24) {
                                                // MobilenetV2, C144H56W56K24S1 but not implement
                                            }
                                        } else if (tempType.channel == 192) {
                                            if (tempType.output_num == 128) {
                                                // YOLO, C192H56W56K128S1 but not implement
                                            }
                                        } else if (tempType.channel == 256) {
                                            if (tempType.output_num == 64) {
                                                if (tempType.batch == 2) { // Resnet, W56C256K64
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {2, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 256) {
                                                // YOLO, C256H56W56K256S1 but not implement
                                            }
                                        }
                                    } else { // width = 112
                                        if (tempType.channel == 16) {
                                            if (tempType.output_num == 96) {
                                                // MobilenetV2, C16H112W112K96S1 but not implement
                                            }
                                        } else if (tempType.channel == 32) {
                                            if (tempType.output_num == 16) {
                                                // MobilenetV2, C32H112W112K16S1 but not implement
                                            } else if (tempType.output_num == 32) {
                                                // MobilenetV2, C32H112W112K32S1 but not implement
                                            } else if (tempType.output_num == 64) {
                                                // Mobilenet, C32H112W112K64S1
                                            }
                                        }
                                    }
                                } else { // stride = 2
                                    if (tempType.width < 14)
                                        continue; // workaround to speed up first
                                    if (tempType.width == 14) {
                                        if ((tempType.channel != 1024
                                             && (tempType.output_num != 512
                                                 || tempType.output_num != 2048)))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 1024) {
                                            if (tempType.output_num == 512) {
                                                if (tempType.batch == 1) { // Resnet, W14C1024K512S2
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 2048) {
                                                if (tempType.batch == 1) { // Resnet,
                                                                           // W14C1024K2048S2
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        }
                                    } else if (tempType.width == 28) {
                                        if ((tempType.channel != 512
                                             && (tempType.output_num != 256
                                                 || tempType.output_num != 1024)))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 512) {
                                            if (tempType.output_num == 256) {
                                                if (tempType.batch == 1) { // Resnet, W28C512K256S2
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {8, 32, 32, 64, 4, 2, 3, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 1024) {
                                                if (tempType.batch == 1) { // Resnet, W28C512K1024S2
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {2, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {2, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        }
                                    } else if (tempType.width == 56) {
                                        if ((tempType.channel != 256
                                             && (tempType.output_num != 128
                                                 || tempType.output_num != 512)))
                                            continue; // workaround to speed up first
                                        if (tempType.channel == 256) {
                                            if (tempType.output_num == 128) {
                                                if (tempType.batch == 1) { // Resnet, W56C256K128S2
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {4, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else if (tempType.batch == 2) {
                                                    tempType.kernel_name = "Conv1x1Atomic.cl";
                                                    tempType.params = {4, 32, 32, 64, 4, 2, 1, 1};
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            } else if (tempType.output_num == 512) {
                                                if (tempType.batch
                                                    <= 2) { // Resnet, C256H56W56K512S2
                                                    tempType.kernel_name =
                                                            "Conv1x1C256H56W56K512S2.cl";
                                                    conv1x1type.push_back(tempType);
                                                } else {
                                                    // todo
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

Conv1x1Type ConvCommon::getKernelInfo(int dev, int stride, int channel, int width, int output_num) {

    init();

    Conv1x1Type mType;
    for (int i = 0; i < conv1x1type.size(); i++) {
        mType.isValid = false;
        if (conv1x1type[i].dev == dev && conv1x1type[i].stride == stride
            && conv1x1type[i].channel == channel && conv1x1type[i].width == width
            && conv1x1type[i].output_num == output_num) {
            mType.isValid = true;
            mType         = conv1x1type[i];
            break;
        }
    }
    return mType;
}

} // namespace miopen
