/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#ifndef DETECT_HEADER_FILE
#define DETECT_HEADER_FILE

#include <iomanip>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "settings.h"
#include "benchmark.h"
#include "detection_postprocess.hpp"

using namespace std;
using namespace tflite;
using namespace CK;


template<typename TData, typename TInConverter, typename TOutConverter>
class TFLiteBenchmark : public Benchmark<TData, TInConverter, TOutConverter> {
public:
    TFLiteBenchmark(Settings *settings, Interpreter *interpreter, int input_index)
            : Benchmark<TData, TInConverter, TOutConverter>(
            settings, interpreter->typed_tensor<TData>(input_index),
            interpreter->typed_output_tensor<TData>(0),
            interpreter->typed_output_tensor<TData>(1),
            interpreter->typed_output_tensor<TData>(2),
            interpreter->typed_output_tensor<TData>(3)) {
    }
};


Settings settings;
BenchmarkSession session(&settings);

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    ops::custom::detection_postprocess::OpData* result;

    result=(ops::custom::detection_postprocess::OpData*)ops::custom::detection_postprocess::Init(context, buffer, length);

    if (!settings.default_model_settings()) {
        result->use_regular_non_max_suppression = true;

        if (settings.get_max_detections() < 0) {
            settings.set_max_detections(result->max_detections);
        } else {
            result->max_detections = settings.get_max_detections();
        }

        if (settings.get_max_classes_per_detection() < 0) {
            settings.set_max_classes_per_detection(result->max_classes_per_detection);
        } else {
            result->max_classes_per_detection = settings.get_max_classes_per_detection();
        }

        if (settings.get_detections_per_class() < 0) {
            if (result->detections_per_class <= 0) {
                result->detections_per_class = ops::custom::detection_postprocess::kNumDetectionsPerClass;
            }
            settings.set_detections_per_class(result->detections_per_class);
        } else {
            result->detections_per_class = settings.get_detections_per_class();
        }

        if (settings.get_num_classes() < 0) {
            settings.set_num_classes(result->num_classes);
        } else {
            result->num_classes = settings.get_num_classes();
        }

        if (settings.get_nms_score_threshold() < 0) {
            settings.set_nms_score_threshold(result->non_max_suppression_score_threshold);
        } else {
            result->non_max_suppression_score_threshold = settings.get_nms_score_threshold();
        }

        if (settings.get_nms_iou_threshold() < 0) {
            settings.set_nms_iou_threshold(result->intersection_over_union_threshold);
        } else {
            result->intersection_over_union_threshold = settings.get_nms_iou_threshold();
        }

        if (settings.get_y_scale() < 0) {
            settings.set_y_scale(result->scale_values.y);
        } else {
            result->scale_values.y = settings.get_y_scale();
        }

        if (settings.get_x_scale() < 0) {
            settings.set_x_scale(result->scale_values.x);
        } else {
            result->scale_values.x = settings.get_x_scale();
        }

        if (settings.get_h_scale() < 0) {
            settings.set_h_scale(result->scale_values.h);
        } else {
            result->scale_values.h = settings.get_h_scale();
        }

        if (settings.get_w_scale() < 0) {
            settings.set_w_scale(result->scale_values.w);
        } else {
            result->scale_values.w = settings.get_w_scale();
        }

        context->AddTensors(context, 1, &result->decoded_boxes_index);
        context->AddTensors(context, 1, &result->scores_index);
        context->AddTensors(context, 1, &result->active_candidate_index);

        //result = op_data;
    } else {
        settings.set_max_detections(result->max_detections);
        settings.set_max_classes_per_detection(result->max_classes_per_detection);
        settings.set_detections_per_class(result->detections_per_class);
        settings.set_num_classes(result->num_classes);
        settings.set_nms_score_threshold(result->non_max_suppression_score_threshold);
        settings.set_nms_iou_threshold(result->intersection_over_union_threshold);
        settings.set_y_scale(result->scale_values.y);
        settings.set_x_scale(result->scale_values.x);
        settings.set_h_scale(result->scale_values.h);
        settings.set_w_scale(result->scale_values.w);
    }

    if (settings.full_report()) {
        std::cout << "-----------Model parameters-----------" << std::endl;
        std::cout << "max_detections: " << result->max_detections << std::endl;
        std::cout << "max_classes_per_detection: " << result->max_classes_per_detection << std::endl;
        std::cout << "detections_per_class: " << result->detections_per_class << std::endl;
        std::cout << "use_regular_non_max_suppression: " << result->use_regular_non_max_suppression << std::endl;
        std::cout << "nms_score_threshold: " << result->non_max_suppression_score_threshold
                  << std::endl;
        std::cout << "nms_iou_threshold: " << result->intersection_over_union_threshold << std::endl;
        std::cout << "num_classes: " << result->num_classes << std::endl;
        std::cout << "y_scale: " << result->scale_values.y << std::endl;
        std::cout << "x_scale: " << result->scale_values.x << std::endl;
        std::cout << "h_scale: " << result->scale_values.h << std::endl;
        std::cout << "w_scale: " << result->scale_values.w << std::endl;
        std::cout << "--------------------------------------" << std::endl;
    }

    return result;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    session.measure_begin();
    ops::custom::detection_postprocess::Eval(context,node);
    session.measure_end_non_max_suppression();
    return kTfLiteOk;
}

TfLiteRegistration* Register_Postprocess_with_NMS() {

    static TfLiteRegistration r = {Init,
                                   ops::custom::detection_postprocess::Free,
                                   ops::custom::detection_postprocess::Prepare,
                                   Eval};
    return &r;
}

#endif