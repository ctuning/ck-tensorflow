/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

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
#include <cwctype>
#include <locale>

#include <xopenme.h>

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
        X_VAR_TIME_GRAPH_AVG,
        X_VAR_TIME_GRAPH_TOTAL,

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

        float avg() const { return _count > 0 ? _total / static_cast<float>(_count): 0.0f; }

    private:
        float _total = 0;
        int _count = 0;
    };

//----------------------------------------------------------------------

    class BenchmarkSession {
    public:
        BenchmarkSession(Settings *settings) {
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
        /// Only one stage can be measured at a time, unless an alternative timer is provided.
        void measure_begin(std::chrono::time_point<std::chrono::high_resolution_clock> *start_time=NULL) {
            auto now = std::chrono::high_resolution_clock::now();
            if (start_time == NULL) {
                _start_time = now;
            } else {
                *start_time = now;
            }
        }

        /// Finish measuring of batch loading stage
        float measure_end_load_images() {
            float duration = measure_end();
            if (_settings->verbose())
                std::cout << "Batch loaded in " << duration << " s" << std::endl;
            _loading_time.add(duration);
            return duration;
        }

        /// Finish measuring of batch prediction stage
        float measure_end_prediction() {
            float duration = measure_end();
            _total_prediction_time += duration;
            if (_settings->verbose())
                std::cout << "Batch classified in " << duration << " s" << std::endl;
            // Skip first batch in order to account warming-up the system
            if (_batch_index > 0 || _settings->batch_count() == 1)
                _prediction_time.add(duration);
            return duration;
        }

        /// Finish measuring of non_max_suppression stage
        float measure_end_non_max_suppression(std::chrono::time_point<std::chrono::high_resolution_clock> *start_time=NULL) {
            float duration = measure_end(start_time);
            _total_prediction_time += duration;
            if (_settings->verbose())
                std::cout << "non_max_suppression completed in " << duration << " s" << std::endl;
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
        Settings *_settings;
        float _total_prediction_time = 0;
        std::vector<FileInfo> _batch_files;
        std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

        float measure_end(std::chrono::time_point<std::chrono::high_resolution_clock> *start_time=NULL) const {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed;
            if (start_time == NULL) {
                elapsed = now - _start_time;
            } else {
                elapsed = now - *start_time;
            }
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
        store_value_f(X_VAR_TIME_IMG_LOAD_TOTAL, "load_images_time_total_s", s.total_load_images_time());
        store_value_f(X_VAR_TIME_IMG_LOAD_AVG, "load_images_time_avg_s", s.avg_load_images_time());
        store_value_f(X_VAR_TIME_CLASSIFY_TOTAL, "prediction_time_total_s", s.total_prediction_time());
        store_value_f(X_VAR_TIME_CLASSIFY_AVG, "prediction_time_avg_s", s.avg_prediction_time());
        store_value_f(X_VAR_TIME_NON_MAX_SUPPRESSION_TOTAL, "non_max_suppression_time_total_s", s.total_non_max_suppression_time());
        store_value_f(X_VAR_TIME_NON_MAX_SUPPRESSION_AVG, "non_max_suppression_time_avg_s", s.avg_non_max_suppression_time());
        store_value_f(X_VAR_TIME_GRAPH_AVG, "graph_time_avg_s", s.avg_prediction_time() - s.avg_non_max_suppression_time());
        store_value_f(X_VAR_TIME_GRAPH_TOTAL, "graph_time_total_s", s.total_prediction_time() - s.total_non_max_suppression_time());

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
        ImageData(Settings *s) : StaticBuffer(
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
        ResultData(Settings *s) : _size(s->detections_buffer_size()) {
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
        virtual ~IBenchmark() {}

        virtual void load_images(const std::vector<FileInfo> &batch_images) = 0;

        virtual void save_results(const std::vector<FileInfo> &batch_images) = 0;

        virtual void export_results(const std::vector<FileInfo> &batch_images) = 0;
    };


    template<typename TData, typename TInConverter, typename TOutConverter>
    class Benchmark : public IBenchmark {
    public:
        Benchmark(Settings *settings,
                  TData *in_ptr,
                  TData *boxes_ptr,
                  TData *classes_ptr,
                  TData *scores_ptr,
                  TData *num_ptr): _settings(settings) {
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

        void export_results(const std::vector<FileInfo> &batch_images) override {
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
        Settings *_settings;
    };

//----------------------------------------------------------------------

    class InCopy {
    public:
        InCopy(Settings *s) {}

        void convert(const ImageData *source, uint8_t *target) const {
            std::copy(source->data(), source->data() + source->size(), target);
        }
    };

//----------------------------------------------------------------------

    class InNormalize {
    public:
        InNormalize(Settings *s) :
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

    void box_to_output(float x1,
                       float y1,
                       float x2,
                       float y2,
                       float score,
                       int detected_class,
                       std::string &buffer,
                       std::vector<std::string> &model_classes,
                       bool correct_background) {
        int class_id_add = correct_background ? 1 : 0;

        std::ostringstream stringStream;
        std::string class_name = detected_class < model_classes.size()
                                 ? model_classes[detected_class]
                                 : "unknown";
        stringStream << std::setprecision(2)
                     << std::showpoint << std::fixed
                     << x1 << " " << y1 << " "
                     << x2 << " " << y2 << " "
                     << std::setprecision(3)
                     << score << " " << detected_class + class_id_add << " "
                     << class_name;
        buffer = stringStream.str();
    }

    class OutCopy {
    public:
        OutCopy(Settings *s): _settings(s) {}

        void convert(const float *boxes,
                     const float *classes,
                     const float *scores,
                     const float *num,
                     ResultData *target,
                     FileInfo src,
                     std::vector<std::string> model_classes,
                     bool correct_background) const {
            std::string *buffer = target->data();
            buffer[0] = to_string(src.width) + " " + to_string(src.height);

            for (int i = 0; i < *num; i++) {
                float y1 = boxes[i * 4] * src.height;
                float x1 = boxes[i * 4 + 1] * src.width;
                float y2 = boxes[i * 4 + 2] * src.height;
                float x2 = boxes[i * 4 + 3] * src.width;
                float score = scores[i];
                int detected_class = int(classes[i]);

                box_to_output(x1, y1, x2, y2, score, detected_class, buffer[i+1], model_classes, correct_background);
            }

            for (int i = *num + 1; i < _settings->detections_buffer_size(); i++) buffer[i] = "";
        }
    private:
        Settings *_settings;
    };

//----------------------------------------------------------------------

    class OutDequantize {
    public:
        OutDequantize(Settings *s): _settings(s) {}

        void convert(const uint8_t *boxes,
                     const uint8_t *classes,
                     const uint8_t *scores,
                     const uint8_t *num,
                     ResultData *target,
                     FileInfo src,
                     std::vector<std::string> model_classes,
                     bool correct_background) const {
            std::string *buffer = target->data();
            buffer[0] = to_string(src.width) + " " + to_string(src.height);
            if (*num == 0) return;

            for (int i = 0; i < *num; i++) {
                float y1 = boxes[i * sizeof(float)] * src.height / 255.0f;
                float x1 = boxes[i * sizeof(float) + 1] * src.width / 255.0f;
                float y2 = boxes[i * sizeof(float) + 2] * src.height / 255.0f;
                float x2 = boxes[i * sizeof(float) + 3] * src.width / 255.0f;
                float score = scores[i] / 255.0f;
                int detected_class = int(classes[i]);

                box_to_output(x1, y1, x2, y2, score, detected_class, buffer[i+1], model_classes, correct_background);
            }

            for (int i = *num + 1; i < _settings->detections_buffer_size(); i++) buffer[i] = "";
        }
    private:
        Settings *_settings;
    };

} // namespace CK

#endif
