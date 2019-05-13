/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */


#ifndef NON_MAX_SUPRESSION_H
#define NON_MAX_SUPRESSION_H

#include <math.h>
#include <vector>

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

inline void swap_float(float *a, float *b) { float c = *a; *a = *b; *b = c;}

inline void swap_int(int &a, int &b) { int c = a; a = b; b = c;}

inline void swap_int(int *a, int *b) { int c = *a; *a = *b; *b = c;}


// Check if `box` intersects (x1, y1, x2, y2) greater than `threshold` of area
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
        int class_id,
        float threshold) {
    for (int i = 0; i < detection_boxes.size(); i++) {
        if (is_box_hidden_by_other(detection_boxes[i], x1, y1, x2, y2, threshold)) return;
    }
    detection_boxes.push_back({x1, y1, x2, y2, score, class_id});
}

void postprocess_detections(Settings *s,
                            std::vector<DetectionBox> &detection_boxes,
                            const float *in_boxes,
                            const float *in_scores,
                            int src_width,
                            int src_height,
                            bool correct_background) {
    float *scores = s->get_scores_sorting_buf();
    int *classes = s->get_classes_sorting_buf();
    int *boxes = s->get_boxes_sorting_buf();
    int *classes_ids = s->get_classes_ids_sorting_buf();

    float *new_scores = scores + s->get_max_total_detections();
    int *new_classes = classes + s->get_max_total_detections();
    int *new_boxes = boxes + s->get_max_total_detections();

    for (int i = 0; i< s->get_max_total_detections(); i++) {
        scores[i] = 0.0f;
    }

    for (int num_box = 0; num_box < s->get_anchors_count(); num_box++) {
        for (int i = 0; i < s->get_max_classes_per_detection(); i++) {
            new_scores[i] = 0.0f;
            new_classes[i] = 0;
            new_boxes[i] = num_box;
        }
        for (int i = 0; i < s->get_num_classes(); i++) {
            classes_ids[i] = i;
        }

        for (int counter = 0; counter < s->get_max_classes_per_detection(); counter++)
        {
            float max_score = in_scores[num_box * s->get_num_classes() + counter + 1];
            int max_class_id = classes_ids[counter + 1];

            for (int num_class = counter + 1; num_class < s->get_num_classes(); num_class++) {
                int index = num_box * s->get_num_classes() + num_class;
                if (max_score < in_scores[index]) {
                    swap_float((float*)in_scores + num_box * s->get_num_classes() + counter + 1, (float*)in_scores + index);
                    swap_int(classes_ids + counter + 1, classes_ids + num_class);
                    max_score = in_scores[num_box * s->get_num_classes() + counter + 1];
                    max_class_id = classes_ids[counter + 1];
                }
            }
            if (max_score > s->get_nms_score_threshold()) {
                new_scores[counter] = max_score;
                new_classes[counter] = max_class_id;
            } else {
                break;
            }
        }

        for (int i = 0; i < s->get_max_total_detections() + s->get_max_classes_per_detection() - 1; i++) {
            float max_score = scores[i];
            for (int j = i + 1; j < s->get_max_total_detections() + s->get_max_classes_per_detection(); j++) {
                if (max_score < scores[j]) {
                    swap_float(scores[i], scores[j]);
                    swap_int(classes[i], classes[j]);
                    swap_int(boxes[i], boxes[j]);
                    max_score = scores[i];
                }
            }
        }
    }

    detection_boxes.reserve(s->get_max_detections());

    int add_class_id = correct_background ? -1 : 0;
    float *anchors = s->get_anchors();

    for (int i = 0; i < s->get_max_total_detections(); i++) {
        if (scores[i] < s->get_nms_score_threshold()) break;

        int index = boxes[i] * 4;

        float ay = anchors[index];
        float ax = anchors[index + 1];
        float ah = anchors[index + 2];
        float aw = anchors[index + 3];

        float ty = in_boxes[index];
        float tx = in_boxes[index + 1];
        float th = in_boxes[index + 2];
        float tw = in_boxes[index + 3];

        float x = (tx / s->get_scale_x()) * aw  + ax;
        float y = (ty / s->get_scale_y()) * ah  + ay;
        float w = expf(tw / s->get_scale_w()) * aw;
        float h = expf(th / s->get_scale_h()) * ah;

        float x1 = x - w * 0.5f;
        float x2 = x + w * 0.5f;
        float y1 = y - h * 0.5f;
        float y2 = y + h * 0.5f;

        if (y1 > y2) swap_float(y1, y2);
        if (x1 > x2) swap_float(x1, x2);
        if (y2 < 0.0f || x2 < 0.0f || y1 > 1.0f || x1 > 1.0f) continue;
        if (y1 < 0.0f) y1 = 0.0f;
        if (x1 < 0.0f) x1 = 0.0f;
        if (y2 > 1.0f) y2 = 1.0f;
        if (x2 > 1.0f) x2 = 1.0f;

        add_detection_to_vector(detection_boxes,
                                x1 * src_width,
                                y1 * src_height,
                                x2 * src_width,
                                y2 * src_height,
                                scores[i],
                                classes[i] + add_class_id,
                                s->get_nms_iou_threshold());

        if (detection_boxes.size() == s->get_max_total_detections()) break;
    }
}

#endif //NON_MAX_SUPRESSION_H
