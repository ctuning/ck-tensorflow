//
// Created by ivan on 4/10/19.
//

#ifndef NMS_NON_MAX_SUPPRESSION_H
#define NMS_NON_MAX_SUPPRESSION_H

#define NMS_MAX_DETECTIONS 100
#define NMS_SCORE_THRESHOLD 0.3f
#define NMS_IOU_THRESHOLD 0.6f


#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <math.h>

using namespace tflite;

struct DetectionBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};

inline float max(float a, float b) { return a > b ? a : b; }

inline float min(float a, float b) { return a > b ? b : a; }

inline void swap_float(float &a, float &b) { float c = a; a = b; b = c;}

inline void swap_int(int &a, int &b) { int c = a; a = b; b = c;}

// Check if `box` intersects (x1, y1, x2, y2) greater than `treshold` of area
bool is_box_hidden_by_other(DetectionBox box,
                            float x1,
                            float y1,
                            float x2,
                            float y2,
                            float threshold) {
    float n1 = max(box.x1, x1);
    float n2 = min(box.x2, x2);
    float m1 = max(box.y1, y1);
    float m2 = min(box.y2, y2);
    if (n1 > n2 || m1 > m2) return false;

    float intersection_area = (n2 - n1) * (m2 - m1);
    float box_area = (x2 - x1) * (y2 - y1);
    float main_box_area = (box.x2 - box.x1) * (box.y2 - box.y1);
    float iou = intersection_area / (box_area + main_box_area - intersection_area);
    if (iou < threshold) return false;

    return true;
}

// Analog of tf.image.non_max_suppression
void add_detection_to_vector(
        std::vector<DetectionBox> &detection_boxes,
        float x1,
        float y1,
        float x2,
        float y2,
        float score,
        int class_id) {
    if (y2 < 0.0f || x2 < 0.0f || y1 > 1.0f || x1 > 1.0f) return;
    if (y1 < 0.0f) y1 = 0.0f;
    if (x1 < 0.0f) x1 = 0.0f;
    if (y2 > 1.0f) y2 = 1.0f;
    if (x2 > 1.0f) x2 = 1.0f;
    for (int i = 0; i < detection_boxes.size(); i++) {
        if (is_box_hidden_by_other(detection_boxes[i], x1, y1, x2, y2, NMS_IOU_THRESHOLD)) return;
    }
    detection_boxes.push_back({x1, y1, x2, y2, score, class_id});
}

TfLiteStatus NMS_Prepare(TfLiteContext *context, TfLiteNode *node) {
    using namespace tflite;
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);  //boxes, scores, anchors
    int out_tensors_count = 4;
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), out_tensors_count);

    TfLiteTensor **output = new TfLiteTensor*[out_tensors_count];
    for (int i=0; i < out_tensors_count; i++) {
        output[i] = GetOutput(context, node, i);
    }

    int num_dims[] = {4*NMS_MAX_DETECTIONS, NMS_MAX_DETECTIONS, NMS_MAX_DETECTIONS, 1};
    int count = sizeof(num_dims) / sizeof(*num_dims);

    for (int i = 0; i < count; ++i) {
        TfLiteIntArray *output_size = TfLiteIntArrayCreate(1);
        output_size->data[0] = num_dims[i];
        if (context->ResizeTensor(context, output[i], output_size) != kTfLiteOk) {
            return kTfLiteError;
        }
    }

    return kTfLiteOk;
}

TfLiteStatus NMS_Eval(TfLiteContext *context, TfLiteNode *node) {

    const int in_tensors_count = 3;
    const int out_tensors_count = 4;

    const TfLiteTensor **input = new const TfLiteTensor*[in_tensors_count];
    int in_sizes[3];
    // 0 - boxes, 1 - scores, 2 - anchors
    for (int i = 0; i < in_tensors_count; i++) {
        input[i] = GetInput(context, node, i);
        in_sizes[i] = 1;
        int dimensions = NumDimensions(input[i]);

        for (int j = 0; j < dimensions; j++) {
            in_sizes[i] *= input[i]->dims->data[j];
        }
    }

    TfLiteTensor **output = new TfLiteTensor*[out_tensors_count];
    int out_sizes[4];
    // 0 - boxes, 1 - scores, 2 - classes, 3 - num detections
    for (int i = 0; i < out_tensors_count; i++) {
        output[i] = GetOutput(context, node, i);
        out_sizes[i] = 1;
        int dimensions = NumDimensions(output[i]);

        for (int j = 0; j < dimensions; j++) {
            out_sizes[i] *= output[i]->dims->data[j];
        }
    }

    int box_count = in_sizes[0] / 4;
    int classes_count = in_sizes[1] / box_count;
    float *in_boxes = input[0]->data.f;
    float *in_scores = input[1]->data.f;
    float *anchors = input[2]->data.f;
    float *out_boxes = output[0]->data.f;
    float *out_classes = output[1]->data.f;
    float *out_scores = output[2]->data.f;
    float *out_num_detections = output[3]->data.f;

    float *scores = new float[box_count];
    int *classes = new int[box_count];
    int *box_nums = new int[box_count];
    for (int i = 0; i < box_count; i++) {
        //Normalize probabilities for box
//        float min = in_scores[i*classes_count];
//        float sum = 0;
//        for (int j = 0; j < classes_count; j++) {
//            int index = i*classes_count + j;
//            if (min > in_scores[index]) {
//                min = in_scores[index];
//            }
//            sum += in_scores[index];
//        }
//        sum -= min * classes_count;
//        for (int j = 0; j < classes_count; j++) {
//            int index = i*classes_count + j;
//            in_scores[index] = (in_scores[index] - min) / sum;
//        }

        // Find most probable class
        int index_max = 1;
        float max = in_scores[i*classes_count + 1];
        for (int j = 2; j < classes_count; j++) {
            int index = i*classes_count + j;
            if (max < in_scores[index]) {
                max = in_scores[index];
                index_max = j;
            }
        }
        if (max > NMS_SCORE_THRESHOLD) {
            scores[i] = max;
            classes[i] = index_max;
        } else {
            scores[i] = in_scores[i*classes_count]; // better 0.0f for speed up
            classes[i] = 0;
        }
        box_nums[i] = i;
    }

    std::vector<DetectionBox> detection_boxes;
    detection_boxes.reserve(NMS_MAX_DETECTIONS);

    // Sort detections by descending probability
    // and add to resulting array
    for (int i = 0; i < box_count; i++) {
        // TODO: change for fast sorting
        float max = scores[i];
        int index = i;
        for (int j = i + 1; j < box_count; j++) {
            if (scores[j] > max) {
                index = j;
                max = scores[j];
            }
        }
        if (max < NMS_SCORE_THRESHOLD) break;
        if (i != index) {
            swap_float(scores[i], scores[index]);
            swap_int(classes[i], classes[index]);
            swap_int(box_nums[i], box_nums[index]);
        }

        if (classes[i] == 0) continue; //skip background
        int shift = box_nums[i] * 4;

        float ay = anchors[shift];
        float ax = anchors[shift + 1];
        float ah = anchors[shift + 2];
        float aw = anchors[shift + 3];

        float ty = in_boxes[shift];
        float tx = in_boxes[shift + 1];
        float th = in_boxes[shift + 2];
        float tw = in_boxes[shift + 3];

        float x = (tx / 10.0f) * aw  + ax;
        float y = (ty / 10.0f) * ah  + ay;
        float w = expf(tw / 5.0f) * aw;
        float h = expf(th / 5.0f) * ah;

        float x1 = x - w / 2.0f;
        float x2 = x + w / 2.0f;
        float y1 = y - h / 2.0f;
        float y2 = y + h / 2.0f;

        if (y1 > y2) swap_float(y1, y2);
        if (x1 > x2) swap_float(x1, x2);
	//std::cout << "c: " << classes[i] << "  p: " << scores[i] << std::endl;
        add_detection_to_vector(detection_boxes, x1, y1, x2, y2, scores[i], classes[i]);

        if (detection_boxes.size() == NMS_MAX_DETECTIONS) break;
    }

    for (int i = 0; i < detection_boxes.size(); i++) {
        out_boxes[i * 4] = detection_boxes[i].y1;
        out_boxes[i * 4 + 1] = detection_boxes[i].x1;
        out_boxes[i * 4 + 2] = detection_boxes[i].y2;
        out_boxes[i * 4 + 3] = detection_boxes[i].x2;
        out_classes[i] = float(detection_boxes[i].class_id - 1); //shift for "background" class
        out_scores[i] = detection_boxes[i].score;
    }

    out_num_detections[0] = detection_boxes.size();

    delete(scores);
    delete(classes);
    delete(box_nums);
    return kTfLiteOk;
}

TfLiteRegistration *Register_NMS() {
    static TfLiteRegistration r = {nullptr, nullptr, NMS_Prepare, NMS_Eval};
    return &r;
}

#endif //NMS_NON_MAX_SUPPRESSION_H
