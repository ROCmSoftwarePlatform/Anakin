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
#include "saber/funcs/deformable_conv.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/x86_utils.h"
#endif

using namespace anakin::saber;

#ifdef USE_CUDA
#define DEVICE NV
#define HOST   NVHX86
#endif

#ifdef AMD_GPU
#define DEVICE AMD
#define HOST   AMDHX86
#endif

#define CHECK_RESULT
//#define CHECK_SPEED

#define RUN_BASIC_TEST true

template <typename out_dtype = float, typename in_dtype = out_dtype>
out_dtype deformable_im2col_bilinear(
        const in_dtype* bottom_data,
        const int height,
        const int width,
        float h,
        float w)
{
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high;
    int w_high;

    if (h_low >= height - 1) {
        h_high = h_low = height - 1;
        h = (float) h_low;
    } else {
        h_high = h_low + 1;
    }

    if (w_low >= width - 1) {
        w_high = w_low = width - 1;
        w = (float) w_low;
    } else {
        w_high = w_low + 1;
    }

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;

    out_dtype v1 = bottom_data[h_low * width + w_low];
    out_dtype v2 = bottom_data[h_low * width + w_high];
    out_dtype v3 = bottom_data[h_high * width + w_low];
    out_dtype v4 = bottom_data[h_high * width + w_high];

    float w1 = hh * hw;
    float w2 = hh * lw;
    float w3 = lh * hw;
    float w4 = lh * lw;

    out_dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

template<typename targetType, typename out_dtype = float, typename in_dtype = out_dtype>
void deformable_conv_basic_check(
        Tensor<targetType> &tensor_in,
        Tensor<targetType> &tensor_offset,
        Tensor<targetType> &tensor_out,
        const in_dtype *weights,
        const out_dtype *bias,
        int group,
        int kernel_w,
        int kernel_h,
        int stride_w,
        int stride_h,
        int dilation_w,
        int dilation_h,
        int pad_w,
        int pad_h,
        bool flag_bias,
        float beta = 0.f)
{
    auto src_data     = reinterpret_cast<const in_dtype*>(tensor_in.data());
    auto offset_data  = reinterpret_cast<const in_dtype*>(tensor_offset.data());
    auto dst_data_ref = reinterpret_cast<out_dtype*>(tensor_out.mutable_data());
    auto weights_data = weights;
    bool with_bias = flag_bias;
    auto bias_data = bias;

    int in_num = tensor_out.num();
    int out_channels = tensor_out.channel();
    int out_h = tensor_out.height();
    int out_w = tensor_out.width();

    int in_channel = tensor_in.channel();
    int in_h = tensor_in.height();
    int in_w = tensor_in.width();
    int out_c_group = out_channels / group;
    int in_c_group = in_channel / group;

#pragma omp parallel for num_threads(8) collapse(5) schedule(static)
    for (int n = 0; n < in_num; ++n)
    {
        for (int g = 0; g < group; ++g)
        {
            for (int oc = 0; oc < out_c_group; ++oc)
            {
                for (int oh = 0; oh < out_h; ++oh)
                {
                    for (int ow = 0; ow < out_w; ++ow)
                    {
                        int out_idx = n * group * out_c_group * out_h * out_w + g * out_c_group * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;
                        float bias_d = with_bias ? (float)(bias_data[g * out_c_group + oc]) : 0.0f;
                        dst_data_ref[out_idx] = bias_d + dst_data_ref[out_idx] * beta;

                        for (int ic = 0; ic < in_c_group; ++ic)
                        {
                            for (int kh = 0; kh < kernel_h; ++kh)
                            {
                                for (int kw = 0; kw < kernel_w; ++kw)
                                {
                                    int data_offset_h_idx = (((((n * group + g) * kernel_h + kh) * kernel_w + kw) * 2    ) * out_h + oh) * out_w + ow;
                                    int data_offset_w_idx = (((((n * group + g) * kernel_h + kh) * kernel_w + kw) * 2 + 1) * out_h + oh) * out_w + ow;

                                    float offset_h = offset_data[data_offset_h_idx];
                                    float offset_w = offset_data[data_offset_w_idx];

                                    // TODO: Keep going
                                    float ih = oh * stride_h - pad_h + kh * (dilation_h) + offset_h;
                                    float iw = ow * stride_w - pad_w + kw * (dilation_w) + offset_w;

                                    if (ih < 0 || ih >= in_h) continue;
                                    if (iw < 0 || iw >= in_w) continue;

                                    int tidx = n * in_channel * in_h * in_w
                                               + g * in_c_group * in_h * in_w
                                               + ic * in_h * in_w;

                                    in_dtype val = deformable_im2col_bilinear(&src_data[tidx], in_h, in_w, ih, iw);

                                    int widx = g * out_c_group * in_c_group * kernel_h * kernel_w
                                               + oc * in_c_group * kernel_h * kernel_w
                                               + ic * kernel_h * kernel_w
                                               + kh * kernel_w
                                               + kw;

                                    dst_data_ref[out_idx] += val * (out_dtype)weights_data[widx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


template<typename dtype,typename TargetType_D,typename TargetType_H>
void deformable_conv_cpu_func(const std::vector<Tensor<TargetType_H>*>& input,
                    std::vector<Tensor<TargetType_H>*>& output,
                    DeformableConvParam<TargetType_D>& param) {
    int group = param.group;
    int input_num = input[0]->num();
    int input_channel = input[0]->channel();
    int input_height = input[0]->height();
    int input_width = input[0]->width();
    int output_channel = output[0]->channel();
    int output_height = output[0]->height();
    int output_width = output[0]->width();
    int stride_h = param.stride_h;
    int stride_w = param.stride_w;
    int dilation_h = param.dilation_h;
    int dilation_w = param.dilation_w;
    int pad_h = param.pad_h;
    int pad_w = param.pad_w;
    int kernel_h = param.weight()->height();
    int kernel_w = param.weight()->width();
    bool bias_term = param.bias()->valid_size() > 0;

    Tensor<TargetType_H> weights_host;
    Tensor<TargetType_H> bias_host;
    weights_host.re_alloc(param.weight()->valid_shape(), AK_FLOAT);
    weights_host.copy_from(*(param.weight()));
    bias_host.re_alloc(param.bias()->valid_shape(), AK_FLOAT);
    bias_host.copy_from(*(param.bias()));

    const dtype* bias_ptr = bias_term ? (const float*)bias_host.data() : nullptr;

    deformable_conv_basic_check<TargetType_H>(
        *input[0],
        *input[1],
        *output[0],
        (const dtype*)weights_host.data(),
        bias_ptr,
        group,
        kernel_w,
        kernel_h,
        stride_w,
        stride_h,
        dilation_w,
        dilation_h,
        pad_w,
        pad_h,
        bias_term);
}

TEST(TestSaberFunc, test_saber_conv_results) {
    TestSaberBase<DEVICE, HOST, AK_FLOAT, DeformableConv, DeformableConvParam> testbase_amd(2, 1);
    Env<DEVICE>::env_init();
    Env<HOST>::env_init();

    std::vector<int> kernel_h_v {1, 3};
    std::vector<int> kernel_w_v{1, 3};
    std::vector<int> pad_h_v{0, 1};
    std::vector<int> pad_w_v{0, 1};
    std::vector<int> stride_h_v{1, 2};
    std::vector<int> stride_w_v{1, 2};
    std::vector<int> dilation_h_v{1};
    std::vector<int> dilation_w_v{1};
    std::vector<int> group_v{1};
    std::vector<int> in_h_v{7, 13, 64};
    std::vector<int> in_w_v{7, 13, 64};
    std::vector<int> input_num_v{1};
    std::vector<int> input_channels_v{2, 7, 8};
    std::vector<int> output_channels_v{2, 7, 8};
    std::vector<bool> bias_term_v{true, false};
    std::vector<bool> with_relu_v{true, false};

    if (RUN_BASIC_TEST) {
    for (int bias_term : bias_term_v)
    for (int with_relu : with_relu_v)
    for (auto kernel_h : kernel_h_v)
    for (auto kernel_w : kernel_w_v)
    for (auto pad_h : pad_h_v)
    for (auto pad_w : pad_w_v)
    for (auto stride_h : stride_h_v)
    for (auto stride_w : stride_w_v)
    for (auto dilation_h : dilation_h_v)
    for (auto dilation_w : dilation_w_v)
    for (auto bias_term : bias_term_v)
    for (auto in_channels : input_channels_v)
    for (auto out_channels : output_channels_v)
    for (auto group : group_v) {

        Shape weights_s({out_channels, in_channels, kernel_h, kernel_w}, Layout_NCHW);
        Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);

        Tensor<DEVICE> weights_dev;
        Tensor<DEVICE> bias_dev;

        weights_dev.re_alloc(weights_s, AK_FLOAT);
        fill_tensor_rand(weights_dev, 1.0f, -1.0f);
        if (bias_term) {
            bias_dev.re_alloc(bias_s, AK_FLOAT);
            fill_tensor_rand(bias_dev, 5.0f, -5.0f);
        }
        DeformableConvParam<DEVICE> param_amd(group, pad_h, pad_w,
                                 stride_h, stride_w,
                                 dilation_h, dilation_w,
                                 &weights_dev, &bias_dev);

        for (auto input_num : input_num_v)
        {
            for (auto height : in_h_v)
            {
                for (auto width : in_w_v) {
                    LOG(INFO) << " N "  << input_num    << " C "  << in_channels << " H "  << height   << " W "<< width
                              << " K "  << out_channels << " Y "  << kernel_h    << " X "  << kernel_w
                              << " PH " << pad_h        << " PW " << pad_w
                              << " SH " << stride_h     << " SW " << stride_w
                              << " DH " << dilation_h   << " DH " << dilation_w
                              << " G "  << group        << " Bi " << bias_term;

                    Shape input_shape({input_num,in_channels, height,width}, Layout_NCHW);
                    Shape output_shape = conv_compute_shape(input_shape, param_amd);
                    Shape offset_shape({input_num, group*2*kernel_h*kernel_w, output_shape.height(), output_shape.width()}, Layout_NCHW);

                    Tensor<DEVICE> input_buf;
                    Tensor<DEVICE> offset_buf;

                    input_buf.re_alloc(input_shape, AK_FLOAT);
                    offset_buf.re_alloc(offset_shape, AK_FLOAT);

                    fill_tensor_rand(input_buf, 1.0f, -1.0f);
                    fill_tensor_rand(offset_buf, 3.0f, -3.0f);

                    std::vector<Tensor<DEVICE>*> input_v{&input_buf, &offset_buf};

                    testbase_amd.add_custom_input(input_v);
                    testbase_amd.set_param(param_amd);//set param
                    testbase_amd.run_test(deformable_conv_cpu_func<float, DEVICE, HOST>, 1e-3);//run test
                }
            }
        }
    }
    } // RUN_BASIC_TEST
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
