/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string.h>
#include <vector>
#include <map>

#include <xopenme.h>

#include "coco.hpp"

#define TFLITE_MAX_DETECTIONS 10
#define OUT_BUFFER_SIZE 11
#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace CK {

    enum _TIMERS {
        X_TIMER_SETUP,
        X_TIMER_TEST,

        X_TIMER_COUNT
    };

    enum _VARS {
        X_VAR_TIME_SETUP,
        X_VAR_TIME_TEST,
        X_VAR_TIME_IMG_LOAD_TOTAL,
        X_VAR_TIME_IMG_LOAD_AVG,
        X_VAR_TIME_CLASSIFY_TOTAL,
        X_VAR_TIME_CLASSIFY_AVG,
        X_VAR_TIME_NON_MAX_SUPPRESSION_TOTAL,
        X_VAR_TIME_NON_MAX_SUPPRESSION_AVG,

        X_VAR_COUNT
    };

/// Store named value into xopenme variable.
    inline void store_value_f(int index, const char *name, float value) {
        char *json_name = new char[strlen(name) + 6];
        sprintf(json_name, "\"%s\":%%f", name);
        xopenme_add_var_f(index, json_name, value);
        delete[] json_name;
    }

/// Dummy `sprintf` like formatting function using std::string.
/// It uses buffer of fixed length so can't be used in any cases,
/// generally use it for short messages with numeric arguments.
    template<typename ...Args>
    inline std::string format(const char *str, Args ...args) {
        char buf[1024];
        sprintf(buf, str, args...);
        return std::string(buf);
    }

//----------------------------------------------------------------------

    class Accumulator {
    public:
        void reset() { _total = 0, _count = 0; }

        void add(float value) { _total += value, _count++; }

        float total() const { return _total; }

        float avg() const { return _total / static_cast<float>(_count); }

    private:
        float _total = 0;
        int _count = 0;
    };

//----------------------------------------------------------------------
    struct FileInfo {
        std::string name;
        int width;
        int height;
    };

    struct DetectionBox {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int class_id;
    };

    template<char delimiter>
    class WordDelimitedBy : public std::string {
    };

    template<char delimiter>
    std::istream &operator>>(std::istream &is, WordDelimitedBy<delimiter> &output) {
        std::getline(is, output, delimiter);
        return is;
    }

    inline float max(float a, float b) { return a > b ? a : b; }

    inline float min(float a, float b) { return a > b ? b : a; }

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
    void add_element_to_box(
            std::vector<DetectionBox> &detection_boxes,
            float x1,
            float y1,
            float x2,
            float y2,
            float score,
            int class_id,
            int width,
            int height) {
        if (y2 < 0.0f || x2 < 0.0f || y1 > height || x1 > width) return;
        if (y1 < 0.0f) y1 = 0.0f;
        if (x1 < 0.0f) x1 = 0.0f;
        if (y2 > height) y2 = height;
        if (x2 > width) x2 = width;
        for (int i = 0; i < detection_boxes.size(); i++) {
            if (is_box_hidden_by_other(detection_boxes[i], x1, y1, x2, y2, 0.5)) return;
        }
        detection_boxes.push_back({x1, y1, x2, y2, score, class_id});
    }

//----------------------------------------------------------------------

    class BenchmarkSettings {
    public:
        BenchmarkSettings() {
            //Load settings
            std::ifstream settings_file("env.ini");
            if (!settings_file)
                throw "Unable to open 'env.ini' file";
            std::map<std::string, std::string> settings_from_file;
            for (std::string s; !getline(settings_file, s).fail();) {
                std::cout << s << std::endl;
                std::istringstream iss(s);
                std::vector<std::string> row((std::istream_iterator<WordDelimitedBy<'='>>(iss)),
                                             std::istream_iterator<WordDelimitedBy<'='>>());
                if (row.size() == 1)
                    settings_from_file.emplace(row[0], "");
                else
                    settings_from_file.emplace(row[0], row[1]);
            }
            std::string model_dataset_type = settings_from_file["MODEL_DATASET_TYPE"];
            if (model_dataset_type == "coco") {
                _model_classes = COCO_CLASSES;
            } else {
                throw ("Unsupported model dataset type: " + model_dataset_type);
            }
            _graph_file = settings_from_file["MODEL_TFLITE_GRAPH"];
            _images_dir = settings_from_file["PREPROCESS_OUT_DIR"];
            _images_file = settings_from_file["PREPROCESSED_FILES"];
            _number_of_threads = std::stoi(settings_from_file["NUMBER_OF_PROCESSORS"]);
            _batch_count = std::stoi(settings_from_file["IMAGE_COUNT"]);
            _batch_size = std::stoi(settings_from_file["BATCH_SIZE"]);
            _image_size_height = std::stoi(settings_from_file["MODEL_IMAGE_HEIGHT"]);
            _image_size_width = std::stoi(settings_from_file["MODEL_IMAGE_WIDTH"]);
            _num_channels = std::stoi(settings_from_file["MODEL_IMAGE_CHANNELS"]);
            _correct_background = settings_from_file["MODEL_NEED_BACKGROUND_CORRECTION"] == "True";
            _normalize_img = settings_from_file["MODEL_NORMALIZE_DATA"] == "True";
            _subtract_mean = settings_from_file["MODEL_SUBTRACT_MEAN"] == "True";
            _full_report = settings_from_file["FULL_REPORT"] == "True";
            _detections_out_dir = settings_from_file["DETECTIONS_OUT_DIR"];
            _use_neon = settings_from_file["USE_NEON"] == "True";
            _use_opencl = settings_from_file["USE_OPENCL"] == "True";
            _verbose = settings_from_file["VERBOSE"] == "True";
            // Print settings
            if (_verbose || _full_report) {
                std::cout << "Graph file: " << _graph_file << std::endl;
                std::cout << "Image dir: " << _images_dir << std::endl;
                std::cout << "Image list: " << _images_file << std::endl;
                std::cout << "Image size: " << _image_size_width << "*" << _image_size_height << std::endl;
                std::cout << "Image channels: " << _num_channels << std::endl;
                std::cout << "Result dir: " << _detections_out_dir << std::endl;
                std::cout << "Batch count: " << _batch_count << std::endl;
                std::cout << "Batch size: " << _batch_size << std::endl;
                std::cout << "Normalize: " << _normalize_img << std::endl;
                std::cout << "Subtract mean: " << _subtract_mean << std::endl;
                std::cout << "Use NEON: " << _use_neon << std::endl;
                std::cout << "Use OPENCL: " << _use_opencl << std::endl;
            }

            // Create results dir if none
            auto dir = opendir(_detections_out_dir.c_str());
            if (dir)
                closedir(dir);
            else
                system(("mkdir " + _detections_out_dir).c_str());

            // Load list of images to be processed
            std::ifstream file(_images_file);
            if (!file)
                throw "Unable to open image list file " + _images_file;
            for (std::string s; !getline(file, s).fail();) {
                std::istringstream iss(s);
                std::vector<std::string> row((std::istream_iterator<WordDelimitedBy<';'>>(iss)),
                                             std::istream_iterator<WordDelimitedBy<';'>>());
                FileInfo fileInfo = {row[0], std::stoi(row[1]), std::stoi(row[2])};
                _image_list.emplace_back(fileInfo);
            }

            if (_verbose || _full_report) {
                std::cout << "Image count in file: " << _image_list.size() << std::endl;
            }
        }

        const std::vector<FileInfo> &image_list() const { return _image_list; }

        const std::vector<std::string> &model_classes() const { return _model_classes; }

        int batch_count() { return _batch_count; }

        int batch_size() { return _batch_size; }

        int image_size_height() { return _image_size_height; }

        int image_size_width() { return _image_size_width; }

        int num_channels() { return _num_channels; }

        int number_of_threads() { return _number_of_threads; }

        bool correct_background() { return _correct_background; }

        bool full_report() { return _full_report; }

        bool normalize_img() { return _normalize_img; }

        bool subtract_mean() { return _subtract_mean; }

        bool use_neon() { return _use_neon; }

        bool use_opencl() { return _use_opencl; }

        bool verbose() { return _verbose; };

        std::string graph_file() { return _graph_file; }

        std::string images_dir() { return _images_dir; }

        std::string detections_out_dir() { return _detections_out_dir; }

    private:
        std::string _detections_out_dir;
        std::string _graph_file;
        std::string _images_dir;
        std::string _images_file;

        std::vector<FileInfo> _image_list;
        std::vector<std::string> _model_classes;

        int _batch_count;
        int _batch_size;
        int _image_size_height;
        int _image_size_width;
        int _num_channels;
        int _number_of_threads;

        bool _correct_background;
        bool _full_report;
        bool _normalize_img;
        bool _subtract_mean;
        bool _use_neon;
        bool _use_opencl;
        bool _verbose;
    };

//----------------------------------------------------------------------

    class BenchmarkSession {
    public:
        BenchmarkSession(BenchmarkSettings *settings) {
            _settings = settings;
        }

        virtual ~BenchmarkSession() {}

        float total_load_images_time() const { return _loading_time.total(); }

        float total_prediction_time() const { return _total_prediction_time; }

        float total_non_max_suppression_time() const { return _non_max_suppression_time.total(); }

        float avg_load_images_time() const { return _loading_time.avg(); }

        float avg_prediction_time() const { return _prediction_time.avg(); }

        float avg_non_max_suppression_time() const { return _non_max_suppression_time.avg(); }

        bool get_next_batch() {
            if (_batch_index + 1 == _settings->batch_count())
                return false;
            _batch_index++;
            int batch_number = _batch_index + 1;
            if (_settings->full_report() || batch_number % 10 == 0)
                std::cout << "\nBatch " << batch_number << " of " << _settings->batch_count() << std::endl;
            int begin = _batch_index * _settings->batch_size();
            int end = (_batch_index + 1) * _settings->batch_size();
            int images_count = _settings->image_list().size();
            if (begin >= images_count || end > images_count)
                throw format("Not enough images to populate batch %d", _batch_index);
            _batch_files.clear();
            for (int i = begin; i < end; i++)
                _batch_files.emplace_back(_settings->image_list()[i]);
            return true;
        }

        /// Begin measuring of new benchmark stage.
        /// Only one stage can be measured at a time.
        void measure_begin() {
            _start_time = std::chrono::high_resolution_clock::now();
        }

        /// Finish measuring of batch loading stage
        float measure_end_load_images() {
            float duration = measure_end();
            if (_settings->full_report() || _settings->verbose())
                std::cout << "Batch loaded in " << duration << " s" << std::endl;
            _loading_time.add(duration);
            return duration;
        }

        /// Finish measuring of batch prediction stage
        float measure_end_prediction() {
            float duration = measure_end();
            _total_prediction_time += duration;
            if (_settings->full_report() || _settings->verbose())
                std::cout << "Batch classified in " << duration << " s" << std::endl;
            // Skip first batch in order to account warming-up the system
            if (_batch_index > 0 || _settings->batch_count() == 1)
                _prediction_time.add(duration);
            return duration;
        }

        /// Finish measuring of non_max_suppression stage
        float measure_end_non_max_suppression() {
            float duration = measure_end();
            _total_prediction_time += duration;
            if (_settings->full_report() || _settings->verbose())
                std::cout << "non_max_suppression completed in " << duration << " s" << std::endl;
            // Skip first batch in order to account warming-up the system
            if (_batch_index > 0 || _settings->batch_count() == 1)
                _non_max_suppression_time.add(duration);
            return duration;
        }

        int batch_index() const { return _batch_index; }

        const std::vector<FileInfo> &batch_files() const { return _batch_files; }

    private:
        int _batch_index = -1;
        Accumulator _loading_time;
        Accumulator _prediction_time;
        Accumulator _non_max_suppression_time;
        BenchmarkSettings *_settings;
        float _total_prediction_time = 0;
        std::vector<FileInfo> _batch_files;
        std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

        float measure_end() const {
            auto finish_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish_time - _start_time;
            return static_cast<float>(elapsed.count());
        }
    };

//----------------------------------------------------------------------

    inline void init_benchmark() {
        xopenme_init(X_TIMER_COUNT, X_VAR_COUNT);
    }

    inline void finish_benchmark(const BenchmarkSession &s) {
        // Store metrics
        store_value_f(X_VAR_TIME_SETUP, "setup_time_s", xopenme_get_timer(X_TIMER_SETUP));
        store_value_f(X_VAR_TIME_TEST, "test_time_s", xopenme_get_timer(X_TIMER_TEST));
        store_value_f(X_VAR_TIME_IMG_LOAD_TOTAL, "images_load_time_s", s.total_load_images_time());
        store_value_f(X_VAR_TIME_IMG_LOAD_AVG, "images_load_time_avg_s", s.avg_load_images_time());
        store_value_f(X_VAR_TIME_CLASSIFY_TOTAL, "prediction_time_total_s", s.total_prediction_time());
        store_value_f(X_VAR_TIME_CLASSIFY_AVG, "prediction_time_avg_s", s.avg_prediction_time());
        store_value_f(X_VAR_TIME_NON_MAX_SUPPRESSION_TOTAL, "non_max_suppression_time_total_s", s.total_non_max_suppression_time());
        store_value_f(X_VAR_TIME_NON_MAX_SUPPRESSION_AVG, "non_max_suppression_avg_s", s.avg_non_max_suppression_time());

        // Finish xopenmp
        xopenme_dump_state();
        xopenme_finish();
    }

    template<typename L>
    void measure_setup(L &&lambda_function) {
        xopenme_clock_start(X_TIMER_SETUP);
        lambda_function();
        xopenme_clock_end(X_TIMER_SETUP);
    }

    template<typename L>
    void measure_prediction(L &&lambda_function) {
        xopenme_clock_start(X_TIMER_TEST);
        lambda_function();
        xopenme_clock_end(X_TIMER_TEST);
    }

//----------------------------------------------------------------------

    template<typename TData>
    class StaticBuffer {
    public:
        StaticBuffer(int size, const std::string &dir) : _size(size), _dir(dir) {
            _buffer = new TData[size];
        }

        virtual ~StaticBuffer() {
            delete[] _buffer;
        }

        TData *data() const { return _buffer; }

        int size() const { return _size; }

    protected:
        const int _size;
        const std::string _dir;
        TData *_buffer;
    };

//----------------------------------------------------------------------

    class ImageData : public StaticBuffer<uint8_t> {
    public:
        ImageData(BenchmarkSettings *s) : StaticBuffer(
                s->image_size_height() * s->image_size_width() * s->num_channels(), s->images_dir()) {}

        void load(const std::string &filename) {
            auto path = filename;
            std::ifstream file(path, std::ios::in | std::ios::binary);
            if (!file) throw "Failed to open image data " + path;
            file.read(reinterpret_cast<char *>(_buffer), _size);
        }
    };

//----------------------------------------------------------------------

    class ResultData {
    public:
        ResultData(BenchmarkSettings *s) : _size(OUT_BUFFER_SIZE) {
            _buffer = new std::string[_size];
        }

        ~ResultData() {
            delete[] _buffer;
        }

        void save(const std::string &filename) {
            std::ofstream file(filename);
            if (!file) throw "Unable to create result file " + filename;
            for (int i = 0; i < _size; i++) {
                if (_buffer[i].length() == 0) break;
                file << _buffer[i] << std::endl;
            }
        }

        int size() const { return _size; }


        std::string *data() const { return _buffer; }

    private:
        std::string *_buffer;
        const int _size;
    };

//----------------------------------------------------------------------

    class IBenchmark {
    public:
        virtual void load_images(const std::vector<FileInfo> &batch_images) = 0;

        virtual void save_results(const std::vector<FileInfo> &batch_images) = 0;

        virtual void non_max_suppression(const std::vector<FileInfo> &batch_images) = 0;
    };


    template<typename TData, typename TInConverter, typename TOutConverter>
    class Benchmark : public IBenchmark {
    public:
        Benchmark(BenchmarkSettings *settings,
                  TData *in_ptr,
                  TData *boxes_ptr,
                  TData *classes_ptr,
                  TData *scores_ptr,
                  TData *num_ptr) {
            _settings = settings;
            _in_ptr = in_ptr;
            _boxes_ptr = boxes_ptr;
            _classes_ptr = classes_ptr;
            _scores_ptr = scores_ptr;
            _num_ptr = num_ptr;
            _in_data.reset(new ImageData(settings));
            _out_data.reset(new ResultData(settings));
            _in_converter.reset(new TInConverter(settings));
            _out_converter.reset(new TOutConverter(settings));
        }

        void load_images(const std::vector<FileInfo> &batch_images) override {
            int image_offset = 0;
            for (auto image_file : batch_images) {
                std::string file_name = _settings->images_dir() + "/" + image_file.name;
                _in_data->load(file_name);
                _in_converter->convert(_in_data.get(), _in_ptr + image_offset);
                image_offset += _in_data->size();
            }
        }

        void non_max_suppression(const std::vector<FileInfo> &batch_images) override {
            int offset = 0;
            int size = _out_data->size();
            for (auto image_file : batch_images) {
                _out_converter->convert(_boxes_ptr + offset * size * 4,
                                        _classes_ptr + offset * size,
                                        _scores_ptr + offset * size,
                                        _num_ptr + offset,
                                        _out_data.get(),
                                        image_file,
                                        _settings->model_classes(),
                                        _settings->correct_background());
                offset += 1;
            }
        }


        void save_results(const std::vector<FileInfo> &batch_images) override {
            for (auto image_file : batch_images) {
                std::size_t found = image_file.name.find_last_of(".");
                std::string result_name = image_file.name.substr(0, found) + ".txt";
                std::string file_name = _settings->detections_out_dir() + "/" + result_name;
                _out_data->save(file_name);
            }
        }

    private:
        TData *_in_ptr;
        TData *_boxes_ptr;
        TData *_classes_ptr;
        TData *_scores_ptr;
        TData *_num_ptr;
        std::unique_ptr<ImageData> _in_data;
        std::unique_ptr<ResultData> _out_data;
        std::unique_ptr<TInConverter> _in_converter;
        std::unique_ptr<TOutConverter> _out_converter;
        BenchmarkSettings *_settings;
    };

//----------------------------------------------------------------------

    class InCopy {
    public:
        InCopy(BenchmarkSettings *s) {}

        void convert(const ImageData *source, uint8_t *target) const {
            std::copy(source->data(), source->data() + source->size(), target);
        }
    };

//----------------------------------------------------------------------

    class InNormalize {
    public:
        InNormalize(BenchmarkSettings *s) :
                _normalize_img(s->normalize_img()), _subtract_mean(s->subtract_mean()) {
        }

        void convert(const ImageData *source, float *target) const {
            // Copy image data to target
            float sum = 0;
            for (int i = 0; i < source->size(); i++) {
                float px = source->data()[i];
                if (_normalize_img)
                    px = (px / 255.0 - 0.5) * 2.0;
                sum += px;
                target[i] = px;
            }
            // Subtract mean value if required
            if (_subtract_mean) {
                float mean = sum / static_cast<float>(source->size());
                for (int i = 0; i < source->size(); i++)
                    target[i] -= mean;
            }
        }

    private:
        const bool _normalize_img;
        const bool _subtract_mean;
    };

//----------------------------------------------------------------------

    void boxes_info_to_output(std::vector<DetectionBox> detection_boxes,
                              std::string *buffer,
                              std::vector<std::string> model_classes,
                              bool correct_background) {
        int class_id_add = correct_background ? 1 : 0;
        for (int i = 0; i < detection_boxes.size(); i++) {
            std::ostringstream stringStream;
            std::string class_name = detection_boxes[i].class_id < model_classes.size()
                                     ? model_classes[detection_boxes[i].class_id]
                                     : "unknown";
            stringStream << std::setprecision(2)
                         << std::showpoint << std::fixed
                         << detection_boxes[i].x1 << " " << detection_boxes[i].y1 << " "
                         << detection_boxes[i].x2 << " " << detection_boxes[i].y2 << " "
                         << std::setprecision(3)
                         << detection_boxes[i].score << " " << detection_boxes[i].class_id + class_id_add << " "
                         << class_name;
            buffer[i] = stringStream.str();
        }
        for (int i = detection_boxes.size(); i < OUT_BUFFER_SIZE-1; i++) buffer[i] = "";
    }

    class OutCopy {
    public:
        OutCopy(const BenchmarkSettings *s) {}

        void convert(const float *boxes,
                     const float *classes,
                     const float *scores,
                     const float *num,
                     ResultData *target,
                     FileInfo src,
                     std::vector<std::string> model_classes,
                     bool correct_background) const {
            std::string *buffer = target->data();
            buffer[0] = std::to_string(src.width) + " " + std::to_string(src.height);
            if (*num == 0) return;

            std::vector<DetectionBox> detection_boxes = {};

            for (int i = 0; i < *num; i++) {
                float y1 = boxes[i * sizeof(float)] * src.height;
                float x1 = boxes[i * sizeof(float) + 1] * src.width;
                float y2 = boxes[i * sizeof(float) + 2] * src.height;
                float x2 = boxes[i * sizeof(float) + 3] * src.width;
                float score = scores[i];
                int detected_class = int(classes[i]);
                add_element_to_box(detection_boxes, x1, y1, x2, y2, score, detected_class, src.width, src.height);
            }
            boxes_info_to_output(detection_boxes, buffer + 1, model_classes, correct_background);
        }
    };

//----------------------------------------------------------------------

    class OutDequantize {
    public:
        OutDequantize(const BenchmarkSettings *s) {}

        void convert(const uint8_t *boxes,
                     const uint8_t *classes,
                     const uint8_t *scores,
                     const uint8_t *num,
                     ResultData *target,
                     FileInfo src,
                     std::vector<std::string> model_classes,
                     bool correct_background) const {
            std::string *buffer = target->data();
            buffer[0] = std::to_string(src.width) + " " + std::to_string(src.height);
            if (*num == 0) return;

            std::vector<DetectionBox> detection_boxes = {};

            for (int i = 0; i < *num; i++) {
                float y1 = boxes[i * sizeof(float)] * src.height / 255.0f;
                float x1 = boxes[i * sizeof(float) + 1] * src.width / 255.0f;
                float y2 = boxes[i * sizeof(float) + 2] * src.height / 255.0f;
                float x2 = boxes[i * sizeof(float) + 3] * src.width / 255.0f;
                float score = scores[i] / 255.0f;
                int detected_class = int(classes[i]);
                add_element_to_box(detection_boxes, x1, y1, x2, y2, score, detected_class, src.width, src.height);
            }
            boxes_info_to_output(detection_boxes, buffer + 1, model_classes, correct_background);
        }
    };

} // namespace CK
