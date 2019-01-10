/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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
#include "saber/core/context.h"
#include "funcs/eltwise_act.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "tensor_op.h"
#include "saber/core/tensor_op.h"
#include "saber_types.h"
#include <vector>

#include <chrono>

using namespace anakin::saber;

struct ProblemConfig {
    int input_num {1};
    int in_channel {32};
    int height {48};
    int width {28};

    int type {2};
    std::vector<float> coeff {1, 1};

    bool channel_shared {false};

    int slope_n {1};
    int slope_c {32};
    int slope_h {1};
    int slope_w {1};
};

ProblemConfig g_config;

template <typename dtype,typename TargetType_D,typename TargetType_H>
void eltwise_cpu(const std::vector<Tensor<TargetType_H>*>& input,
                 std::vector<Tensor<TargetType_H>*>& output,
                 EltwiseActiveParam<TargetType_D>& param) {

    EltwiseParam<AMD> eltwise_param = param.eltwise_param;

    int num     = input[0]->num();
    int channel = input[0]->channel();
    int height  = input[0]->height();
    int width   = input[0]->width();

    int size    = height * width;
    size_t count = input[0]->valid_size();

    int out_size = output[0]->size();
    int in_size  = input[0]->size();

    const int num_arrs = input.size();

    const dtype* src = (const dtype*)input[0]->data();
    dtype* dst = (dtype*)output[0]->mutable_data();

    switch (eltwise_param.operation) {
        case Eltwise_sum:
            for (int e = 0; e < in_size; e++) {
                dst[e] = eltwise_param.coeff[0] * src[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                src = (const dtype*)input[a]->data();
                for (int e = 0; e < in_size; e++) {
                    dst[e] += eltwise_param.coeff[a] * src[e];
                }
            }
            break;
        case Eltwise_prod:
            for (int e = 0; e < in_size; e++) {
                dst[e] =  src[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                src = (const dtype*)input[a]->data();
                for (int e = 0; e < in_size; e++) {
                    dst[e] *=  src[e];
                }
            }
            break;
        case Eltwise_max:
            for (int e = 0; e < in_size; e++) {
                dst[e] =  src[e];
            }
            for (int a = 1; a < num_arrs; a++) {
                src = (const dtype*)input[a]->data();
                for (int e = 0; e < in_size; e++) {
                    dst[e] =  (dst[e]>src[e])?dst[e]:src[e];
                }
            }
            break;

        default:
           break;
    }

    if(param.activation_param.has_active){
        switch (param.activation_param.active) {
            case Active_relu:
            {
                for (int a = 0; a < num_arrs; a++) {
                    for (int e = 0; e < in_size; e++) {
                       dst[e] =  (dst[e]>0.0f)?dst[e]:0.0f;
                    }
                }
            } break;
            case Active_prelu:
            {
                auto prelu_param  = param.activation_param.prelu_param;

                Tensor<TargetType_D>* slop_dev;
                slop_dev = prelu_param.slope;
                Shape shape = slop_dev->valid_shape();
                Tensor<TargetType_H>* slop_host;//(shape);
                slop_host = new Tensor<TargetType_H>(shape);
                slop_host->copy_from(*slop_dev);
                const dtype* slope_ptr = (const dtype*)slop_host->data();

                for (int n = 0; n < num; n++) {
                    const dtype* in_ptr = dst + n * channel * size;
                    dtype* out_ptr = dst + n * channel * size;

                    for (int c = 0; c < channel; c++) {
                        const dtype* in_ch_ptr = in_ptr + c * size;
                        dtype* out_ch_ptr = out_ptr + c * size;
                        dtype slope = prelu_param.channel_shared ?  slope_ptr[0] : slope_ptr[c];

                        for (int k = 0; k < size; k++) {
                            out_ch_ptr[k] = in_ch_ptr[k] > 0 ? in_ch_ptr[k] : in_ch_ptr[k] * slope;
                        }
                    }
                }

                delete slop_host;
            } break;
            default: {
            } break;
        }
    }
}

EltwiseType getEltwiseType() {
    int type = g_config.type;
    if (2 == type) {
        return Eltwise_sum;
    } else if (3 == type) {
        return Eltwise_max;
    } else if (1 == type) {
        return Eltwise_prod;
    }
    return Eltwise_unknow;
}


TEST(TestSaberFunc, test_func_eltwise){

#ifdef AMD_GPU
    //Init the test_base
    Env<AMD>::env_init();
    TestSaberBase<AMD,AMDHX86,AK_FLOAT,EltwiseActive, EltwiseActiveParam> testbase_amd(2,1);
#endif

    EltwiseType eltwise_type = getEltwiseType();

    for(int num_in:{g_config.input_num}){
        for(int c_in:{g_config.in_channel}){
            for(int h_in:{g_config.height}){
                for(int w_in:{g_config.width}){
                    for(EltwiseType type:{eltwise_type}){
                        LOG(INFO)<<"input = "<<num_in<<", type = "<<type;
                    #ifdef AMD_GPU
                        Shape slope_shape({g_config.slope_n, g_config.slope_c, g_config.slope_h, g_config.slope_w}, Layout_NCHW);
                        Tensor<AMD> slope_tensor;
                        slope_tensor.re_alloc(slope_shape, AK_FLOAT);
                        fill_tensor_rand(slope_tensor, -1.0, 1.0);

                        PreluParam<AMD> prelu_param(g_config.channel_shared, &slope_tensor);
                        ActivationParam<AMD> activation_param(Active_prelu, 0, 0, prelu_param);

                        EltwiseParam<AMD> eltwise_param(type,g_config.coeff);
                        EltwiseActiveParam<AMD> param_amd(eltwise_param, activation_param);

                        testbase_amd.set_param(param_amd);
                        testbase_amd.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
                        testbase_amd.run_test(eltwise_cpu<float, AMD, AMDHX86>);
                    #endif
                    }
                }
            }
        }
    }
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
