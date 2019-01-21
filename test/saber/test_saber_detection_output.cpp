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
#include "saber/funcs/detection_output.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "saber/funcs/impl/detection_helper.h"

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

template <typename dtype>
void decode_bbox_corner_variance(const int count,
                                 const dtype* loc_data, const dtype* prior_data, const dtype* variance,
                                 const int num_priors, const bool share_location, const int num_loc_classes,
                                 const int background_label_id, dtype* bbox_data) {
    for (int index = 0; index < count; ++index) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;

        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            continue;
        }

        //! variance is encoded in target, we simply need to add the offset predictions.
        bbox_data[idx]     = prior_data[idx_p]     + loc_data[idx];
        bbox_data[idx + 1] = prior_data[idx_p + 1] + loc_data[idx + 1];
        bbox_data[idx + 2] = prior_data[idx_p + 2] + loc_data[idx + 2];
        bbox_data[idx + 3] = prior_data[idx_p + 3] + loc_data[idx + 3];
    }
}

template <typename dtype>
void decode_bbox_corner_no_variance(const int count,
                                    const dtype* loc_data, const dtype* prior_data, const dtype* variance,
                                    const int num_priors, const bool share_location, const int num_loc_classes,
                                    const int background_label_id, dtype* bbox_data) {
    for (int index = 0; index < count; ++index) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;

        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            continue;
        }

        //! variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[idx]     = prior_data[idx_p]     + loc_data[idx]     * variance[idx_p];
        bbox_data[idx + 1] = prior_data[idx_p + 1] + loc_data[idx + 1] * variance[idx_p + 1];
        bbox_data[idx + 2] = prior_data[idx_p + 2] + loc_data[idx + 2] * variance[idx_p + 2];
        bbox_data[idx + 3] = prior_data[idx_p + 3] + loc_data[idx + 3] * variance[idx_p + 3];
    }
}

template <typename dtype>
void decode_bbox_center_variance(const int count,
                                 const dtype* loc_data, const dtype* prior_data, const dtype* variance,
                                 const int num_priors, const bool share_location, const int num_loc_classes,
                                 const int background_label_id, dtype* bbox_data) {
    for (int index = 0; index < count; ++index) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;

        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            continue;
        }

        const dtype p_xmin = prior_data[idx_p];
        const dtype p_ymin = prior_data[idx_p + 1];
        const dtype p_xmax = prior_data[idx_p + 2];
        const dtype p_ymax = prior_data[idx_p + 3];

        const dtype prior_width    = p_xmax - p_xmin;
        const dtype prior_height   = p_ymax - p_ymin;
        const dtype prior_center_x = (p_xmin + p_xmax) / 2.;
        const dtype prior_center_y = (p_ymin + p_ymax) / 2.;

        const dtype xmin = loc_data[idx];
        const dtype ymin = loc_data[idx + 1];
        const dtype xmax = loc_data[idx + 2];
        const dtype ymax = loc_data[idx + 3];

        //! variance is encoded in target, we simply need to retore the offset predictions.
        dtype decode_bbox_center_x = xmin * prior_width + prior_center_x;
        dtype decode_bbox_center_y = ymin * prior_height + prior_center_y;
        dtype decode_bbox_width    = exp(xmax) * prior_width;
        dtype decode_bbox_height   = exp(ymax) * prior_height;

        bbox_data[idx + 0] = decode_bbox_center_x - decode_bbox_width / 2.f;
        bbox_data[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
        bbox_data[idx + 2] = decode_bbox_center_x + decode_bbox_width / 2.f;
        bbox_data[idx + 3] = decode_bbox_center_y + decode_bbox_height / 2.f;
    }
}

template <typename dtype>
void decode_bbox_center_no_variance(const int count,
                                    const dtype* loc_data, const dtype* prior_data, const dtype* variance,
                                    const int num_priors, const bool share_location, const int num_loc_classes,
                                    const int background_label_id, dtype* bbox_data) {
    for (int index = 0; index < count; ++index) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;

        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            continue;
        }

        const dtype p_xmin = prior_data[idx_p];
        const dtype p_ymin = prior_data[idx_p + 1];
        const dtype p_xmax = prior_data[idx_p + 2];
        const dtype p_ymax = prior_data[idx_p + 3];

        const dtype prior_width    = p_xmax - p_xmin;
        const dtype prior_height   = p_ymax - p_ymin;
        const dtype prior_center_x = (p_xmin + p_xmax) / 2.;
        const dtype prior_center_y = (p_ymin + p_ymax) / 2.;

        const dtype xmin = loc_data[idx];
        const dtype ymin = loc_data[idx + 1];
        const dtype xmax = loc_data[idx + 2];
        const dtype ymax = loc_data[idx + 3];

        //! variance is encoded in bbox, we need to scale the offset accordingly.
        dtype decode_bbox_center_x = variance[idx_p] * xmin * prior_width + prior_center_x;
        dtype decode_bbox_center_y = variance[idx_p + 1] * ymin * prior_height + prior_center_y;
        dtype decode_bbox_width    = exp(variance[idx_p + 2] * xmax) * prior_width;
        dtype decode_bbox_height   = exp(variance[idx_p + 3] * ymax) * prior_height;

        bbox_data[idx]     = decode_bbox_center_x - decode_bbox_width / 2.f;
        bbox_data[idx + 1] = decode_bbox_center_y - decode_bbox_height / 2.f;
        bbox_data[idx + 2] = decode_bbox_center_x + decode_bbox_width / 2.f;
        bbox_data[idx + 3] = decode_bbox_center_y + decode_bbox_height / 2.f;
    }
}

template <typename dtype>
void decode_bbox_corner_size_variance(const int count,
                                      const dtype* loc_data, const dtype* prior_data, const dtype* variance,
                                      const int num_priors, const bool share_location, const int num_loc_classes,
                                      const int background_label_id, dtype* bbox_data) {
    for (int index = 0; index < count; ++index) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;

        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            continue;
        }

        const dtype p_xmin = prior_data[idx_p];
        const dtype p_ymin = prior_data[idx_p + 1];
        const dtype p_xmax = prior_data[idx_p + 2];
        const dtype p_ymax = prior_data[idx_p + 3];
        const dtype prior_width = p_xmax - p_xmin;
        const dtype prior_height = p_ymax - p_ymin;

        //! variance is encoded in target, we simply need to add the offset predictions.
        bbox_data[idx]     = p_xmin + loc_data[idx] * prior_width;
        bbox_data[idx + 1] = p_ymin + loc_data[idx + 1] * prior_height;
        bbox_data[idx + 2] = p_xmax + loc_data[idx + 2] * prior_width;
        bbox_data[idx + 3] = p_ymax + loc_data[idx + 3] * prior_height;
    }
}

template <typename dtype>
void decode_bbox_corner_size_no_variance(const int count,
        const dtype* loc_data, const dtype* prior_data, const dtype* variance,
        const int num_priors, const bool share_location, const int num_loc_classes,
        const int background_label_id, dtype* bbox_data) {
    for (int index = 0; index < count; ++index) {
        const int c = index % num_loc_classes;
        const int idx_p = (index % num_priors) * 4;
        const int idx = index * 4;

        if (!share_location && c == background_label_id) {
            //! Ignore background class if not share_location.
            continue;
        }

        const dtype p_xmin = prior_data[idx_p];
        const dtype p_ymin = prior_data[idx_p + 1];
        const dtype p_xmax = prior_data[idx_p + 2];
        const dtype p_ymax = prior_data[idx_p + 3];
        const dtype prior_width = p_xmax - p_xmin;
        const dtype prior_height = p_ymax - p_ymin;

        //! variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[idx] = p_xmin + loc_data[idx] * variance[idx_p] * prior_width;
        bbox_data[idx + 1] = p_ymin + loc_data[idx + 1] * variance[idx_p + 1] * prior_height;
        bbox_data[idx + 2] = p_xmax + loc_data[idx + 2] * variance[idx_p + 2] * prior_width;
        bbox_data[idx + 3] = p_ymax + loc_data[idx + 3] * variance[idx_p + 3] * prior_height;
    }
}

template <typename Dtype>
void do_decode_bboxes(const int nthreads,
                      const Dtype* loc_data, const Dtype* prior_data,
                      const CodeType code_type, const bool variance_encoded_in_target,
                      const int num_priors, const bool share_location,
                      const int num_loc_classes, const int background_label_id,
                      Dtype* bbox_data) {
    int count = nthreads / 4;
    const Dtype* variance_data = prior_data + 4 * num_priors;

    if (code_type == CORNER) {
        if (variance_encoded_in_target) {
            decode_bbox_corner_variance<Dtype>
            (count, loc_data, prior_data, variance_data, num_priors, share_location,
             num_loc_classes, background_label_id, bbox_data);
        } else {
            decode_bbox_corner_no_variance<Dtype>
            (count, loc_data, prior_data, variance_data, num_priors, share_location,
             num_loc_classes, background_label_id, bbox_data);
        }
    } else if (code_type == CENTER_SIZE) {
        if (variance_encoded_in_target) {
            decode_bbox_center_variance<Dtype>
            (count, loc_data, prior_data, variance_data, num_priors, share_location,
             num_loc_classes, background_label_id, bbox_data);
        } else {
            decode_bbox_center_no_variance<Dtype>
            (count, loc_data, prior_data, variance_data, num_priors, share_location,
             num_loc_classes, background_label_id, bbox_data);
        }
    } else if (code_type == CORNER_SIZE) {
        if (variance_encoded_in_target) {
            decode_bbox_corner_size_variance<Dtype>
            (count, loc_data, prior_data, variance_data, num_priors, share_location,
             num_loc_classes, background_label_id, bbox_data);
        } else {
            decode_bbox_corner_size_no_variance<Dtype>
            (count, loc_data, prior_data, variance_data, num_priors, share_location,
             num_loc_classes, background_label_id, bbox_data);
        }
    }
}

template <typename dtype>
void permute_data(const int nthreads, const dtype* data, const int num_classes,
                  const int num_data, const int num_dim, dtype* new_data) {
    for (int index = 0; index < nthreads; ++index) {
        const int i = index % num_dim;
        const int c = (index / num_dim) % num_classes;
        const int d = (index / num_dim / num_classes) % num_data;
        const int n = index / num_dim / num_classes / num_data;
        const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;
        new_data[new_index] = data[index];
    }
}


template<typename dtype, typename TargetType_D, typename TargetType_H>
void detection_output_cpu_func(const std::vector<Tensor<TargetType_H>*>& inputs,
                               std::vector<Tensor<TargetType_H>*>& outputs,
                               DetectionOutputParam<TargetType_D>& param) {
    Tensor<HOST>* t_loc   = inputs[0]; // {N, boxes * 4, 1, 1}
    Tensor<HOST>* t_conf  = inputs[1]; // {N, classes * boxes, 1, 1}
    Tensor<HOST>* t_prior = inputs[2]; // {1, 1, 2, boxes * 4(xmin, ymin, xmax, ymax)}

    const int num = t_loc->num();

    int _num_priors = t_prior->valid_size() / 8;
    int _num_classes = param.class_num ? param.class_num : t_conf->valid_size() / (num * _num_priors);
    int _num_loc_classes = param.share_location ? 1 : _num_classes;

    Tensor<HOST> _bbox_permute;
    Tensor<HOST> _bbox_preds;
    Tensor<HOST> _conf_permute;

    _bbox_permute.reshape(t_loc->shape());  // N * class * boxes * 4
    _bbox_preds.reshape(t_loc->shape());    // N * boxes * class * 4
    _conf_permute.reshape(t_conf->shape());

    const dtype* loc_data = static_cast<const dtype*>(t_loc->data());
    const dtype* prior_data = static_cast<const dtype*>(t_prior->data());

    dtype* bbox_data = static_cast<dtype*>(_bbox_preds.mutable_data());
    const int loc_count = _bbox_preds.valid_size();
    do_decode_bboxes<dtype>(loc_count,
                            loc_data,   // input 1 location data
                            prior_data, // input 3 prior boxes
                            param.type,
                            param.variance_encode_in_target,
                            _num_priors,
                            param.share_location,
                            _num_loc_classes,
                            param.background_id,
                            bbox_data);     // decode box data

    // Retrieve all decoded location predictions.
    if (!param.share_location) {
        dtype* bbox_permute_data = static_cast<dtype*>(_bbox_permute.mutable_data());
        permute_data<dtype>(loc_count, bbox_data, _num_loc_classes, _num_priors, 4, bbox_permute_data);
    }

    // Retrieve all confidences.
    dtype* conf_permute_data = static_cast<dtype*>(_conf_permute.mutable_data());
    permute_data<dtype>(t_conf->valid_size(), static_cast<dtype*>(t_conf->data()), _num_classes,
                        _num_priors, 1, conf_permute_data);

    std::vector<dtype> result;
    nms_detect(bbox_data, conf_permute_data, result, num, _num_classes, _num_priors,
               param.background_id,
               param.keep_top_k, param.nms_top_k, param.conf_thresh, param.nms_thresh, param.nms_eta,
               param.share_location);

    if (result.size() == 0) {
        result.resize(7);

        for (int i = 0; i < 7; ++i) {
            result[i] = (dtype) - 1;
        }

        outputs[0]->reshape(Shape({1, 1, 1, 7}));
    } else {
        outputs[0]->reshape(Shape({1, 1, static_cast<int>(result.size() / 7), 7}));
    }

    dtype* output_ptr = static_cast<dtype*>(outputs[0]->mutable_data());
    memcpy(output_ptr, result.data(), result.size() * sizeof(dtype));
}

static void fillCornerInput(
    DetectionOutputParam<DEVICE>& param,
    Tensor<DEVICE>& input_loc,
    Tensor<DEVICE>& input_conf,
    Tensor<DEVICE>& input_box) {
    Tensor<HOST> tmp;

    srand(time(0));

    int _batch_size = input_loc.shape().num();
    int _num_priors = input_box.valid_size() / 8;
    int _num_class  = param.class_num ? param.class_num : input_conf.valid_size() /
                      (_batch_size * _num_priors);

    // fill loc data
    int totalboxes = input_loc.valid_size() / 4;
    tmp.re_alloc(input_loc.shape(), AK_FLOAT);
    float* loc_ptr = static_cast<float*>(tmp.mutable_data());

    for (int i = 0; i < totalboxes; i++) {
        loc_ptr[i * 4 + 0]  = static_cast<float>(rand()) / RAND_MAX * 0.05;
        loc_ptr[i * 4 + 1]  = static_cast<float>(rand()) / RAND_MAX * 0.05;
        loc_ptr[i * 4 + 2]  = static_cast<float>(rand()) / RAND_MAX * 0.05;
        loc_ptr[i * 4 + 3]  = static_cast<float>(rand()) / RAND_MAX * 0.05;
    }

    input_loc.copy_from(tmp);

    // fill conf data
    tmp.re_alloc(input_conf.shape(), AK_FLOAT);
    float* conf_ptr = static_cast<float*>(tmp.mutable_data());

    for (int i = 0; i < (input_conf.valid_size() / _num_class); i++) {
        std::vector<float> score(_num_class);
        float total_score = 0.0f;

        for (int c = 0; c < _num_class; c++) {
            score[c] = static_cast<float>(rand());
            total_score += score[c];
        }

        for (int c = 0; c < _num_class; c++) {
            score[c] = score[c] / total_score;
        }

        for (int c = 0; c < _num_class; c++) {
            conf_ptr[i * _num_class + c] = score[c];
        }
    }

    input_conf.copy_from(tmp);

    // fill box data: prior and varation
    tmp.re_alloc(input_box.shape(), AK_FLOAT);
    float* box_ptr = static_cast<float*>(tmp.mutable_data());
    float* var_ptr = box_ptr + _num_priors * 4;

    for (int i = 0; i < _num_priors; i++) {
        float dx = static_cast<float>(rand()) / RAND_MAX * 0.2 + 0.1;
        float dy = static_cast<float>(rand()) / RAND_MAX * 0.2 + 0.1;
        float x  = static_cast<float>(rand()) / RAND_MAX * 0.7 + 0.15;
        float y  = static_cast<float>(rand()) / RAND_MAX * 0.7 + 0.15;

        box_ptr[i * 4 + 0] = x - dx * 0.5;
        box_ptr[i * 4 + 1] = y - dy * 0.5;
        box_ptr[i * 4 + 2] = x + dx * 0.5;
        box_ptr[i * 4 + 3] = y + dy * 0.5;
    }

    for (int i = 0; i < _num_priors; i++) {
        var_ptr[i * 4 + 0] = static_cast<float>(rand()) / RAND_MAX;
        var_ptr[i * 4 + 1] = static_cast<float>(rand()) / RAND_MAX;
        var_ptr[i * 4 + 2] = static_cast<float>(rand()) / RAND_MAX;
        var_ptr[i * 4 + 3] = static_cast<float>(rand()) / RAND_MAX;
    }

    input_box.copy_from(tmp);
}

static void fillCenterSizeInput(
    DetectionOutputParam<DEVICE>& param,
    Tensor<DEVICE>& input_loc,
    Tensor<DEVICE>& input_conf,
    Tensor<DEVICE>& input_box) {
    Tensor<HOST> tmp;

    srand(time(0));

    int _batch_size = input_loc.shape().num();
    int _num_priors = input_box.valid_size() / 8;
    int _num_class  = param.class_num ? param.class_num : input_conf.valid_size() /
                      (_batch_size * _num_priors);

    // fill loc data
    int totalboxes = input_loc.valid_size() / 4;
    tmp.re_alloc(input_loc.shape(), AK_FLOAT);
    float* loc_ptr = static_cast<float*>(tmp.mutable_data());

    for (int i = 0; i < totalboxes; i++) {
        loc_ptr[i * 4 + 0]  = static_cast<float>(rand()) / RAND_MAX * 0.5;
        loc_ptr[i * 4 + 1]  = static_cast<float>(rand()) / RAND_MAX * 0.5;
        loc_ptr[i * 4 + 2]  = static_cast<float>(rand()) / RAND_MAX * 0.5;
        loc_ptr[i * 4 + 3]  = static_cast<float>(rand()) / RAND_MAX * 0.5;
    }

    input_loc.copy_from(tmp);

    // fill conf data
    tmp.re_alloc(input_conf.shape(), AK_FLOAT);
    float* conf_ptr = static_cast<float*>(tmp.mutable_data());

    for (int i = 0; i < (input_conf.valid_size() / _num_class); i++) {
        std::vector<float> score(_num_class);
        float total_score = 0.0f;

        for (int c = 0; c < _num_class; c++) {
            score[c] = static_cast<float>(rand());
            total_score += score[c];
        }

        for (int c = 0; c < _num_class; c++) {
            score[c] = score[c] / total_score;
        }

        for (int c = 0; c < _num_class; c++) {
            conf_ptr[i * _num_class + c] = score[c];
        }
    }

    input_conf.copy_from(tmp);

    // fill box data: prior and varation
    tmp.re_alloc(input_box.shape(), AK_FLOAT);
    float* box_ptr = static_cast<float*>(tmp.mutable_data());
    float* var_ptr = box_ptr + _num_priors * 4;

    for (int i = 0; i < _num_priors; i++) {
        float dx = static_cast<float>(rand()) / RAND_MAX * 0.2;
        float dy = static_cast<float>(rand()) / RAND_MAX * 0.2;
        float x  = static_cast<float>(rand()) / RAND_MAX * 0.8 + 0.1;
        float y  = static_cast<float>(rand()) / RAND_MAX * 0.8 + 0.1;

        box_ptr[i * 4 + 0] = x - dx * 0.5;
        box_ptr[i * 4 + 1] = y - dy * 0.5;
        box_ptr[i * 4 + 2] = x + dx * 0.5;
        box_ptr[i * 4 + 3] = y + dy * 0.5;
    }

    for (int i = 0; i < _num_priors; i++) {
        var_ptr[i * 4 + 0] = static_cast<float>(rand()) / RAND_MAX * 0.2;
        var_ptr[i * 4 + 1] = static_cast<float>(rand()) / RAND_MAX * 0.2;
        var_ptr[i * 4 + 2] = static_cast<float>(rand()) / RAND_MAX * 0.2;
        var_ptr[i * 4 + 3] = static_cast<float>(rand()) / RAND_MAX * 0.2;
    }

    input_box.copy_from(tmp);

}

static void fillCornerSizeInput(
    DetectionOutputParam<DEVICE>& param,
    Tensor<DEVICE>& input_loc,
    Tensor<DEVICE>& input_conf,
    Tensor<DEVICE>& input_box) {
    Tensor<HOST> tmp;

    srand(time(0));

    int _batch_size = input_loc.shape().num();
    int _num_priors = input_box.valid_size() / 8;
    int _num_class  = param.class_num ? param.class_num : input_conf.valid_size() /
                      (_batch_size * _num_priors);

    // fill loc data
    int totalboxes = input_loc.valid_size() / 4;
    tmp.re_alloc(input_loc.shape(), AK_FLOAT);
    float* loc_ptr = static_cast<float*>(tmp.mutable_data());

    for (int i = 0; i < totalboxes; i++) {
        loc_ptr[i * 4 + 0]  = static_cast<float>(rand()) / RAND_MAX * 0.5;
        loc_ptr[i * 4 + 1]  = static_cast<float>(rand()) / RAND_MAX * 0.5;
        loc_ptr[i * 4 + 2]  = static_cast<float>(rand()) / RAND_MAX * 0.5;
        loc_ptr[i * 4 + 3]  = static_cast<float>(rand()) / RAND_MAX * 0.5;
    }

    input_loc.copy_from(tmp);

    // fill conf data
    tmp.re_alloc(input_conf.shape(), AK_FLOAT);
    float* conf_ptr = static_cast<float*>(tmp.mutable_data());

    for (int i = 0; i < (input_conf.valid_size() / _num_class); i++) {
        std::vector<float> score(_num_class);
        float total_score = 0.0f;

        for (int c = 0; c < _num_class; c++) {
            score[c] = static_cast<float>(rand());
            total_score += score[c];
        }

        for (int c = 0; c < _num_class; c++) {
            score[c] = score[c] / total_score;
        }

        for (int c = 0; c < _num_class; c++) {
            conf_ptr[i * _num_class + c] = score[c];
        }
    }

    input_conf.copy_from(tmp);

    // fill box data: prior and varation
    tmp.re_alloc(input_box.shape(), AK_FLOAT);
    float* box_ptr = static_cast<float*>(tmp.mutable_data());
    float* var_ptr = box_ptr + _num_priors * 4;

    for (int i = 0; i < _num_priors; i++) {
        float dx = static_cast<float>(rand()) / RAND_MAX * 0.2;
        float dy = static_cast<float>(rand()) / RAND_MAX * 0.2;
        float x  = static_cast<float>(rand()) / RAND_MAX * 0.8 + 0.1;
        float y  = static_cast<float>(rand()) / RAND_MAX * 0.8 + 0.1;

        box_ptr[i * 4 + 0] = x - dx * 0.5;
        box_ptr[i * 4 + 1] = y - dy * 0.5;
        box_ptr[i * 4 + 2] = x + dx * 0.5;
        box_ptr[i * 4 + 3] = y + dy * 0.5;
    }

    for (int i = 0; i < _num_priors; i++) {
        var_ptr[i * 4 + 0] = static_cast<float>(rand()) / RAND_MAX;
        var_ptr[i * 4 + 1] = static_cast<float>(rand()) / RAND_MAX;
        var_ptr[i * 4 + 2] = static_cast<float>(rand()) / RAND_MAX;
        var_ptr[i * 4 + 3] = static_cast<float>(rand()) / RAND_MAX;
    }

    input_box.copy_from(tmp);
}

static void generateInput(
    DetectionOutputParam<DEVICE>& param,
    Tensor<DEVICE>& input_loc,
    Tensor<DEVICE>& input_conf,
    Tensor<DEVICE>& input_box) {
    switch (param.type) {
    case CORNER: {
        fillCornerInput(param, input_loc, input_conf, input_box);
    }
    break;

    case CENTER_SIZE: {
        fillCenterSizeInput(param, input_loc, input_conf, input_box);
    }
    break;

    case CORNER_SIZE: {
        fillCornerSizeInput(param, input_loc, input_conf, input_box);
    }
    break;

    default: {
        LOG(INFO) << "Error encode type";
        exit(0);
    }
    }

}

TEST(TestSaberFunc, test_saber_conv_results) {
    TestSaberBase<DEVICE, HOST, AK_FLOAT, DetectionOutput, DetectionOutputParam> testbase_amd(3, 1);
    Env<DEVICE>::env_init();
    Env<HOST>::env_init();

    // DetectionOutputParam parameter
    std::vector<bool>     share_location_v {true, false};
    std::vector<bool>     variance_encode_in_target_v {true, false};
    std::vector<int>      class_num_v {17, 21};
    // some of decoded content uninitialized when "backgournd id == class id && !share_location_v"
    std::vector<int>      background_id_v {0xffff};
    std::vector<int>      keep_top_k_v {200, 150};
    std::vector<CodeType> type_v {CORNER, CENTER_SIZE, CORNER_SIZE};
    std::vector<float>    conf_thresh_v {0, 0.01f};
    std::vector<int>      nms_top_k_v {400, 300};
    std::vector<float>    nms_thresh_v {0.45f, 0.4};
    std::vector<float>    nms_eta_v {1.0f, 0.8};

    // DetectionOutputParam input size
    std::vector<int> num_batch_v {1, 3};
    std::vector<int> num_box_v {8736, 4561};


    for (auto share_location : share_location_v)
        for (auto variance_encode_in_target : variance_encode_in_target_v)
            for (auto class_num : class_num_v)
                for (auto background_id : background_id_v)
                    for (auto keep_top_k : keep_top_k_v)
                        for (auto type : type_v)
                            for (auto conf_thresh : conf_thresh_v)
                                for (auto nms_top_k : nms_top_k_v)
                                    for (auto nms_thresh : nms_thresh_v)
                                        for (auto nms_eta : nms_eta_v)
                                            for (auto num_batch : num_batch_v)
                                                for (auto num_box : num_box_v) {

                                                    LOG(INFO) << "Param: share_location "      << share_location
                                                              << " variance_encode_in_target " << variance_encode_in_target
                                                              << " class_num "                 << class_num
                                                              << " background_id "             << background_id
                                                              << " keep_top_k "                << keep_top_k
                                                              << " type "                      << type
                                                              << " conf_thresh "               << conf_thresh
                                                              << " nms_top_k "                 << nms_top_k
                                                              << " nms_thresh "                << nms_thresh
                                                              << " nms_eta "                   << nms_eta
                                                              << " num_batch "                 << num_batch
                                                              << " num_box "                   << num_box;

                                                    DetectionOutputParam<DEVICE> param_amd(class_num, background_id, keep_top_k, nms_top_k,
                                                                                           nms_thresh, conf_thresh, share_location, variance_encode_in_target, type, nms_eta);

                                                    // set up input content
                                                    Tensor<DEVICE> input_loc;
                                                    Tensor<DEVICE> input_conf;
                                                    Tensor<DEVICE> input_box;

                                                    if (param_amd.share_location) {
                                                        input_loc.re_alloc(Shape({num_batch, num_box, 4, 1}, Layout_NCHW), AK_FLOAT);
                                                    } else {
                                                        input_loc.re_alloc(Shape({num_batch, num_box * class_num, 4, 1}, Layout_NCHW), AK_FLOAT);
                                                    }

                                                    input_conf.re_alloc(Shape({num_batch, num_box, class_num, 1}, Layout_NCHW), AK_FLOAT);
                                                    input_box.re_alloc(Shape({1, 2, num_box * 4, 1}, Layout_NCHW), AK_FLOAT);

                                                    generateInput(param_amd , input_loc, input_conf, input_box);

                                                    // run unit test
                                                    std::vector<Tensor<DEVICE>*> input_v {&input_loc, &input_conf, &input_box};
                                                    testbase_amd.add_custom_input(input_v);
                                                    testbase_amd.set_param(param_amd);//set param
                                                    testbase_amd.run_test(detection_output_cpu_func<float, DEVICE, HOST>, 1e-3);//run test
                                                }
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
