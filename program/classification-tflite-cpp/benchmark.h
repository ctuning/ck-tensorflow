/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */
 
#pragma once

#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string.h>

#include <xopenme.h>

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

  X_VAR_COUNT
};

/// Store named value into xopenme variable
inline void store_value_f(int index, const char* name, float value) {
  char* json_name = new char[strlen(name) + 6];
  sprintf(json_name, "\"%s\":%%f", name);
  xopenme_add_var_f(index, json_name, value);
  delete[] json_name;
}

/// Load mandatory string value from the environment
inline std::string getenv_s(const std::string& name) {
  const char *value = getenv(name.c_str());
  if (!value)
    throw "Required environment variable " + name + " is not set";
  return std::string(value);
}

/// Load mandatory integer value from the environment
inline int getenv_i(const std::string& name) {
  const char *value = getenv(name.c_str());
  if (!value)
    throw "Required environment variable " + name + " is not set";
  return atoi(value);
}

template <typename ...Args>
inline std::string format(const char* str, Args ...args) {
  char buf[1024];
  sprintf(buf, std, args...);
  return std::string(buf);
}

//----------------------------------------------------------------------

class Accumulator {
public:
  void reset() { _total = 0, _count = 0; }
  void add(float value) { _total += value, _count++; }
  float total() const { return _total; }
  float avg() const { _total / static_cast<float>(_count); }
private:
  float _total = 0;
  int _count = 0;
};

//----------------------------------------------------------------------

class BenchmarkSettings {
public:
  const std::string graph_file = getenv_s("RUN_OPT_TFLITE_GRAPH");
  const std::string images_dir = getenv_s("RUN_OPT_IMAGE_DIR");
  const std::string images_file = getenv_s("RUN_OPT_IMAGE_LIST");
  const std::string result_dir = getenv_s("RUN_OPT_RESULT_DIR");
  const int batch_count = getenv_i("CK_BATCH_COUNT");
  const int batch_size = getenv_i("CK_BATCH_SIZE");
  const int image_size = getenv_i("CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH");
  const bool normalize_img = getenv_i("RUN_OPT_NORMALIZE_DATA") == 1;
  const bool subtract_mean = getenv_i("RUN_OPT_SUBTRACT_MEAN") == 1;

  const int NUM_CHANNELS = 3;
  const int NUM_CLASSES = 1001;

  BenchmarkSettings() {
    // Print settings
    std::cout << "Graph file: " << graph_file << std::endl;
    std::cout << "Image dir: " << images_dir << std::endl;
    std::cout << "Image list: " << images_file << std::endl;
    std::cout << "Image size: " << image_size << std::endl;
    std::cout << "Result dir: " << result_dir << std::endl;
    std::cout << "Batch count: " << batch_count << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    // Create results dir if none
    auto dir = opendir(result_dir.c_str());
    if (dir)
      closedir(dir);
    else
      system(("mkdir " + result_dir).c_str());

    // Load list of images to be processed
    ifstream file(images_file);
    if (!file)
      throw "Unable to open image list file " + images_file;
    for (string s; !getline(file, s).fail();)
      _image_list.emplace_back(s);
    cout << "Image count in file: " << _image_list.size() << endl;
  }

  const std::vector<std::string>& image_list() const { return _image_list; }

  const std::vector<std::string> _image_list;
};

//----------------------------------------------------------------------

class BenchmarkSession {
public:
  BenchmarkSession(const BenchmarkSettings* settings): _settings(settings) {
  }

  virtual ~BenchmarkSession() {}

  float total_load_images_time() const { return _loading_time.total(); }
  float total_prediction_time() const { return _total_prediction_time; }
  float avg_load_images_time() const { return _loading_time.avg(); }
  float avg_prediction_time() const { return _prediction_time.avg(); }

  bool get_next_batch() {
    if (_batch_index+1 == settings->batch_count)
      return false;
    _batch_index++;
    auto begin = settings->image_list().begin() + _batch_index * _settings->batch_size;
    auto end = begin + _settings->batch_size;
    std::copy(begin, end, _batch_files.begin(), _batch_files.begin());
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
    std::cout << "Batch loaded in " << duration << " s" << std::endl;
    _loading_time.add(duration);
    return duration;
  }

  /// Finish measuring of batch prediction stage
  float measure_end_prediction() {
    float duration = measure_end();
    _total_prediction_time += duration;
    std::cout << "Batch classified in " << duration << " s" << std::endl;
    // Skip first batch in order to account warming-up the system
    if (_batch_index > 0 || _settings->batch_count == 1)
      _prediction_time.add(duration);
    return duration;
  }

  int batch_index() const { return _batch_index; }
  const std::vector<std::string>& batch_files() const { return _batch_files; }

private:
  int _batch_index = -1;
  Accumulator _loading_time;
  Accumulator _prediction_time;
  const BenchmarkSettings const* _settings;
  float _total_prediction_time = 0;
  std::vector<std::string> _batch_files;
  std::chrono::time_point<chrono::high_resolution_clock> _start_time;

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

inline void finish_benchmark(const BenchmarkSession& s) {
  // Print metrics
  std::cout << "-------------------------------\n";
  std::cout << "Graph loaded in " << xopenme_get_timer(X_TIMER_SETUP) << " s" << std::endl;
  std::cout << "All batches loaded in " << s.total_load_images_time() << " s" << std::endl;
  std::cout << "All batches classified in " << s.total_prediction_time() << " s" << std::endl;
  std::cout << "Average classification time: " << s.avg_prediction_time() << " s" << std::endl;
  std::cout << "-------------------------------\n";

  // Store metrics
  store_value_f(X_VAR_TIME_SETUP, "setup_time_s", xopenme_get_timer(X_TIMER_SETUP));
  store_value_f(X_VAR_TIME_TEST, "test_time_s", xopenme_get_timer(X_TIMER_TEST));
  store_value_f(X_VAR_TIME_IMG_LOAD_TOTAL, "images_load_time_s", s.total_load_images_time());
  store_value_f(X_VAR_TIME_IMG_LOAD_AVG, "images_load_time_avg_s", s.avg_load_images_time());
  store_value_f(X_VAR_TIME_CLASSIFY_TOTAL, "prediction_time_total_s", s.total_prediction_time());
  store_value_f(X_VAR_TIME_CLASSIFY_AVG, "prediction_time_avg_s", s.avg_prediction_time());

  // Finish xopenmp
  xopenme_dump_state();
  xopenme_finish();
}

template <typename L>
void measure_setup(L &&lambda_function) {
  xopenme_clock_start(X_TIMER_SETUP);
  lambda_function();
  xopenme_clock_end(X_TIMER_SETUP);
}

template <typename L>
void measure_prediction(L &&lambda_function) const {
  xopenme_clock_start(X_TIMER_TEST);
  lambda_function();
  xopenme_clock_end(X_TIMER_TEST);
}

//----------------------------------------------------------------------

class ImageData {
  ImageData(const BenchmarkSettings* s): _dir(s->images_dir) {
    _buffer.resize(s->image_size * s->image_size * s->NUM_CHANNELS);
  }
  
  void load(const std::string& filename) {
    auto path = _dir + '/' + filename;
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) throw "Failed to open image data " + path;
    file.read(reinterpret_cast<char*>(data()), size());
  }

  uint8_t* data() const { return _buffer.data(); }
  int size() const { return _buffer.size(); }

private:
  const std::string _dir;
  std::vector<uint8_t> _buffer;
};

//----------------------------------------------------------------------

class ResultData {
  ResultData(const BenchmarkSettings* s): _dir(s->result_dir) {
    _buffer.resize(s->NUM_CLASSES);
  }

  void save(const std::string& filename) {
    auto path = _dir + '/' + filename + ".txt"
    std::ofstream file(path);
    if (!file) throw "Unable to create result file " + path;
    for (auto probe : _buffer)
      file << probe << std::endl;
  }

  float* data() const { return _buffer.data(); }
  int size() const { return _buffer.size(); }

private:
  const std::string _dir;
  std::vector<float> _buffer;
};

//----------------------------------------------------------------------

class InCopy {
public:
  InCopy(const BenchmarkSettings* s) {}
  
  void convert(const ImageData& source, uint8_t* target) const {
    std::copy(source.data(), source.data() + source.size(), target);
  }
};

//----------------------------------------------------------------------

class InNormalize {
public:
  InNormalize(const BenchmarkSettings* s):
    _normalize_img(s->normalize_img), _subtract_mean(s->subtract_mean) {
  }
  
  void convert(const ImageData& source, float* target) const {
    // Copy image data to target
    float sum = 0;
    for (int i = 0; i < source.size(); i++) {
      float px = source.data()[i];
      if (_normalize_img)
        px = (px / 255.0 - 0.5) * 2.0;
      sum += px;
      target[i] = px;
    }
    // Subtract mean value if required
    if (_subtract_mean) {
      float mean = sum / static_cast<float>(source.size());
      for (int i = 0; i < source.size(); i++)
        target[i] -= mean;
    }
  }

private:
  const bool _normalize_img;
  const bool _subtract_mean;
};

//----------------------------------------------------------------------

class OutCopy {
public:
  OutCopy(const BenchmarkSettings* s) {}
  
  void convert(float* source, const ResultData& target) const {
    std::copy(source, source + target.size(), target.data());
  }
};

//----------------------------------------------------------------------

class OutDequantize {
public:
  OutDequantize(const BenchmarkSettings* s) {}
  
  void convert(float* source, const ResultData& target) const {
    for (int i = 0; i < target.size(); i++)
      target.data()[i] = source[i] / 255.0;
  }
};

} // namespace CK
