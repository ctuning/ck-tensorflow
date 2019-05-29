/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#ifndef DETECT_SETTINGS_H
#define DETECT_SETTINGS_H

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <list>
#include <thread>

/// Load mandatory string value from the environment.
inline std::string getenv_s(const std::string& name) {
  const char *value = getenv(name.c_str());
  if (!value) {
    throw std::runtime_error("Required environment variable " + name + " is not set");
  }
  return std::string(value);
}

/// Load mandatory integer value from the environment.
inline int getenv_i(const std::string& name) {
  const char *value = getenv(name.c_str());
  if (!value)
    throw std::runtime_error("Required environment variable " + name + " is not set");
  return std::atoi(value);
}

/// Load mandatory float value from the environment.
inline float getenv_f(const std::string& name) {
  const char *value = getenv(name.c_str());
  if (!value)
    throw std::runtime_error("Required environment variable " + name + " is not set");
  return std::atof(value);
}

template <typename T>
std::string to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str() ;
}

struct FileInfo {
    std::string name;
    int width;
    int height;
};

template<char delimiter>
class WordDelimitedBy : public std::string {
};

template<char delimiter>
std::istream &operator>>(std::istream &is, WordDelimitedBy<delimiter> &output) {
    std::getline(is, output, delimiter);
    return is;
}

inline std::string alter_str(std::string a, std::string b) { return a != "" ? a: b; };
inline std::string alter_str(char *a, std::string b) { return a != nullptr ? a: b; };
inline std::string alter_str(char *a, char *b) { return a != nullptr ? a: b; };
inline int alter_str_i(char *a, int b) { return a != nullptr ? std::atoi(a): b; };
inline int alter_str_i(char *a, char *b) { return std::atoi(a != nullptr ? a: b); };
inline int alter_str_i(std::string a, std::string b) { return std::atoi(a != "" ? a.c_str(): b.c_str()); };
inline float alter_str_f(std::string a, std::string b) { return std::atof(a != "" ? a.c_str(): b.c_str()); };
inline float alter_str_f(char *a, char *b) { return std::atof(a != nullptr ? a: b); };

std::string abs_path(std::string, std::string);
std::string str_to_lower(std::string);
std::string str_to_lower(char *);
bool get_yes_no(std::string);
bool get_yes_no(char *);
std::vector<std::string> *readClassesFile(std::string);

#ifdef OBJECT_DETECTION_ARMNN_TFLITE
float *readAnchorsFile(std::string, int&);
#endif //OBJECT_DETECTION_ARMNN_TFLITE

class Settings {
public:
    Settings() {
      try {
        std::string model_dataset_type = getenv_s("CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE");
        if (model_dataset_type != "coco") {
            throw ("Unsupported model dataset type: " + model_dataset_type);
        }

        std::string nms_type = str_to_lower(alter_str(getenv("USE_NMS"), std::string("regular")));
        _fast_nms = nms_type == "regular" ? false : true;

#ifdef OBJECT_DETECTION_ARMNN_TFLITE
        _graph_file = getenv("CK_ENV_TENSORFLOW_MODEL_TFLITE_GRAPH_NO_NMS");

        std::string anchors_file = abs_path(getenv_s("CK_ENV_TENSORFLOW_MODEL_ROOT"),
                                   getenv_s("CK_ENV_TENSORFLOW_MODEL_ANCHORS"));
        _m_anchors = readAnchorsFile(anchors_file, _m_anchors_count);
        _use_neon = get_yes_no(getenv("USE_NEON"));
        _use_opencl = get_yes_no(getenv("USE_OPENCL"));
        _default_model_settings = false;
#else
        if (nms_type == "regular") {
            _graph_file = getenv_s("CK_ENV_TENSORFLOW_MODEL_TFLITE_GRAPH_REGULAR_NMS");
        } else {
            _graph_file = getenv_s("CK_ENV_TENSORFLOW_MODEL_TFLITE_GRAPH_FAST_NMS");
        }
        
        _number_of_threads = std::thread::hardware_concurrency();
        _number_of_threads = _number_of_threads < 1 ? 1 : _number_of_threads;
        _number_of_threads = alter_str_i(getenv("CK_HOST_CPU_NUMBER_OF_PROCESSORS"), _number_of_threads);
        _default_model_settings=!get_yes_no(getenv("USE_CUSTOM_NMS_SETTINGS"));
#endif //OBJECT_DETECTION_ARMNN_TFLITE

        _graph_file = abs_path(getenv_s("CK_ENV_TENSORFLOW_MODEL_ROOT"), _graph_file);
        
        std::string classes_file = abs_path(getenv_s("CK_ENV_TENSORFLOW_MODEL_ROOT"),
                                            getenv_s("CK_ENV_TENSORFLOW_MODEL_CLASSES"));
        _model_classes = *readClassesFile(classes_file);
        _images_dir = getenv_s("CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_DIR");
        _detections_out_dir = getenv_s("CK_DETECTIONS_OUT_DIR");
        _images_file = getenv_s("CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_SUBSET_FOF");
        _image_size_height = getenv_i("CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT");
        _image_size_width = getenv_i("CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH");
        _num_channels = getenv_i("CK_ENV_TENSORFLOW_MODEL_IMAGE_CHANNELS");
        _correct_background = get_yes_no(getenv("CK_ENV_TENSORFLOW_MODEL_NEED_BACKGROUND_CORRECTION"));
        _normalize_img = get_yes_no(getenv_s("CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA"));
        _subtract_mean = get_yes_no(getenv_s("CK_ENV_TENSORFLOW_MODEL_SUBTRACT_MEAN"));

        _batch_count = alter_str_i(getenv("CK_BATCH_COUNT"), 1);
        _batch_size = alter_str_i(getenv("CK_BATCH_SIZE"), 1);
        _full_report = !get_yes_no(getenv("CK_SILENT_MODE"));
        _verbose = get_yes_no(getenv("VERBOSE"));


        if (_default_model_settings) {
            _m_max_classes_per_detection = 1;
            _m_max_detections = getenv_i("CK_ENV_TENSORFLOW_MODEL_MAX_DETECTIONS");
            _m_detections_per_class = 100;
            _m_num_classes = getenv_i("CK_ENV_TENSORFLOW_MODEL_NUM_CLASSES");
            _m_nms_score_threshold = getenv_f("CK_ENV_TENSORFLOW_MODEL_NMS_SCORE_THRESHOLD");
            _m_nms_iou_threshold = getenv_f("CK_ENV_TENSORFLOW_MODEL_NMS_IOU_THRESHOLD");
            _m_scale_h = getenv_f("CK_ENV_TENSORFLOW_MODEL_SCALE_H");
            _m_scale_w = getenv_f("CK_ENV_TENSORFLOW_MODEL_SCALE_W");
            _m_scale_x = getenv_f("CK_ENV_TENSORFLOW_MODEL_SCALE_X");
            _m_scale_y = getenv_f("CK_ENV_TENSORFLOW_MODEL_SCALE_Y");
        } else {
            _m_max_classes_per_detection = alter_str_i(getenv("MAX_CLASSES_PER_DETECTION"), 1);
            if (_m_max_classes_per_detection > 1 && _fast_nms) {
                std::cout << std::endl << "You can't use USE_NMS=fast and MAX_CLASSES_PER_DETECTION>1 at the same time" << std::endl ;
                exit(-1);
            }

            _m_max_detections = alter_str_i(getenv("MAX_DETECTIONS"), getenv("CK_ENV_TENSORFLOW_MODEL_MAX_DETECTIONS"));

#ifdef OBJECT_DETECTION_ARMNN_TFLITE
            _m_max_total_detections = std::max(100, _m_max_detections);
            _m_num_classes = alter_str_i(getenv("NUM_CLASSES"), getenv("CK_ENV_TENSORFLOW_MODEL_NUM_CLASSES")) + _correct_background;
            _m_detections_per_class = alter_str_i(getenv("DETECTIONS_PER_CLASS"), 100);

            _d_boxes = new float [_m_anchors_count * 4]();
            _d_scores = new float [_m_anchors_count * _m_num_classes];

            _d_scores_sort_buf = new float[_m_max_total_detections + _m_max_classes_per_detection];
            _d_classes_sort_buf = new int[_m_max_total_detections + _m_max_classes_per_detection];
            _d_boxes_sort_buf = new int[_m_max_total_detections + _m_max_classes_per_detection];
            _d_classes_ids_sort_buf = new int[_m_num_classes];
#else
            _m_num_classes = alter_str_i(getenv("NUM_CLASSES"), getenv("CK_ENV_TENSORFLOW_MODEL_NUM_CLASSES"));
#endif //OBJECT_DETECTION_ARMNN_TFLITE

            _m_nms_score_threshold = alter_str_f(getenv("NMS_SCORE_THRESHOLD"), getenv("CK_ENV_TENSORFLOW_MODEL_NMS_SCORE_THRESHOLD"));
            _m_nms_iou_threshold = alter_str_f(getenv("NMS_IOU_THRESHOLD"), getenv("CK_ENV_TENSORFLOW_MODEL_NMS_IOU_THRESHOLD"));
            _m_scale_h = alter_str_f(getenv("SCALE_H"), getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_H"));
            _m_scale_w = alter_str_f(getenv("SCALE_W"), getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_W"));
            _m_scale_x = alter_str_f(getenv("SCALE_X"), getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_X"));
            _m_scale_y = alter_str_f(getenv("SCALE_Y"), getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_Y"));
        }

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
#ifdef OBJECT_DETECTION_ARMNN_TFLITE
            std::cout << "Use Neon: " << _use_neon << std::endl;
            std::cout << "Use OpenCL: " << _use_opencl << std::endl;
#endif //OBJECT_DETECTION_ARMNN_TFLITE
        }

        // Load list of images to be processed
        std::ifstream file(_images_file);
        if (!file)
            throw "Unable to open image list file " + _images_file;
        for (std::string s; !getline(file, s).fail();) {
            std::istringstream iss(s);
            std::vector<std::string> row((std::istream_iterator<WordDelimitedBy<';'>>(iss)),
                                         std::istream_iterator<WordDelimitedBy<';'>>());
            FileInfo fileInfo = {row[0], std::atoi(row[1].c_str()), std::atoi(row[2].c_str())};
            _image_list.emplace_back(fileInfo);
        }

        if (_verbose || _full_report) {
            std::cout << "Image count in file: " << _image_list.size() << std::endl;
        }
      } catch(const std::runtime_error& e) {
        std::cout << "Exception during parameter setup: " << e.what() << std::endl;
        exit(1);
      } catch(const char* msg) {
        std::cout << "Exception message during parameter setup: " << msg << std::endl;
        exit(1);
      } catch(const std::string& s) {
        std::cout << "Exception message during parameter setup: " << s << std::endl;
        exit(1);
      }
    }

#ifdef OBJECT_DETECTION_ARMNN_TFLITE
    ~Settings() {
        delete _d_boxes;
        delete _d_scores;
        delete _d_scores_sort_buf;
        delete _d_classes_sort_buf;
        delete _d_boxes_sort_buf;
        delete _d_classes_ids_sort_buf;
    }

    int get_anchors_count() { return _m_anchors_count; }
    int get_max_total_detections() { return _m_max_total_detections; };

    bool use_neon() { return _use_neon; }
    bool use_opencl() { return _use_opencl; }

    float* get_anchors() { return _m_anchors; };
    float* get_boxes_buf() { return _d_boxes; };
    float* get_scores_buf() { return _d_scores; };
    float* get_scores_sorting_buf() { return _d_scores_sort_buf; };

    int* get_classes_sorting_buf() { return _d_classes_sort_buf; }
    int* get_classes_ids_sorting_buf() { return _d_classes_ids_sort_buf; }
    int* get_boxes_sorting_buf() { return _d_boxes_sort_buf; }
#endif //OBJECT_DETECTION_ARMNN_TFLITE

    const std::vector<FileInfo> &image_list() const { return _image_list; }

    const std::vector<std::string> &model_classes() const { return _model_classes; }

    int batch_count() { return _batch_count; }

    int batch_size() { return _batch_size; }

    int detections_buffer_size() { return _m_max_detections + 1; }

    int image_size_height() { return _image_size_height; }

    int image_size_width() { return _image_size_width; }

    int num_channels() { return _num_channels; }

    int number_of_threads() { return _number_of_threads; }

    bool correct_background() { return _correct_background; }

    bool default_model_settings() { return _default_model_settings; }

    bool fast_nms() { return _fast_nms; }

    bool full_report() { return _full_report || _verbose; }

    bool normalize_img() { return _normalize_img; }

    bool subtract_mean() { return _subtract_mean; }

    bool verbose() { return _verbose; };

    int get_max_detections() { return _m_max_detections; };
    void set_max_detections(int i) { _m_max_detections = i;}

    int get_max_classes_per_detection() { return _m_max_classes_per_detection; };
    void set_max_classes_per_detection(int i) { _m_max_classes_per_detection = i;}

    int get_detections_per_class() { return _m_detections_per_class; };
    void set_detections_per_class(int i) { _m_detections_per_class = i;}

    int get_num_classes() { return _m_num_classes; };
    void set_num_classes(int i) { _m_num_classes = i;}

    float get_nms_score_threshold() { return _m_nms_score_threshold; };
    void set_nms_score_threshold(float i) { _m_nms_score_threshold = i;}

    float get_nms_iou_threshold() { return _m_nms_iou_threshold; };
    void set_nms_iou_threshold(float i) { _m_nms_iou_threshold = i;}

    float get_scale_h() { return _m_scale_h; };
    void set_scale_h(float i) { _m_scale_h = i;}

    float get_scale_w() { return _m_scale_w; };
    void set_scale_w(float i) { _m_scale_w = i;}

    float get_scale_x() { return _m_scale_x; };
    void set_scale_x(float i) { _m_scale_x = i;}

    float get_scale_y() { return _m_scale_y; };
    void set_scale_y(float i) { _m_scale_y = i;}

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

#ifdef OBJECT_DETECTION_ARMNN_TFLITE
    int *_d_classes_sort_buf;
    int *_d_boxes_sort_buf;
    int *_d_classes_ids_sort_buf;

    int _m_max_total_detections;
    int _m_anchors_count;

    float *_d_boxes;
    float *_d_scores;
    float *_d_scores_sort_buf;
    float *_m_anchors;

    bool _use_neon;
    bool _use_opencl;
#endif //OBJECT_DETECTION_ARMNN_TFLITE

    int _batch_count;
    int _batch_size;
    int _image_size_height;
    int _image_size_width;
    int _num_channels;
    int _number_of_threads;
    int _m_max_classes_per_detection;
    int _m_max_detections;
    int _m_detections_per_class;
    int _m_num_classes;

    float _m_nms_score_threshold;
    float _m_nms_iou_threshold;
    float _m_scale_h;
    float _m_scale_w;
    float _m_scale_x;
    float _m_scale_y;

    bool _correct_background;
    bool _default_model_settings;
    bool _fast_nms;
    bool _full_report;
    bool _normalize_img;
    bool _subtract_mean;
    bool _verbose;
};

std::vector<std::string> *readClassesFile(std::string filename) {
    std::vector<std::string> *lines = new std::vector<std::string>;
    lines->clear();
    std::ifstream file(filename);
    std::string s;
    while (getline(file, s))
        lines->push_back(s);

    return lines;
}

bool get_yes_no(std::string answer) {
    std::locale loc;
    for (std::string::size_type i=0; i<answer.length(); ++i)
        answer[i] = std::tolower(answer[i],loc);
    if (answer == "1" || answer == "yes" || answer == "on" || answer == "true") return true;
    return false;
}
bool get_yes_no(char *answer) {
    if (answer == nullptr) return false;
    return get_yes_no(std::string(answer));
}

std::string str_to_lower(std::string answer) {
    std::locale loc;
    for (std::string::size_type i=0; i<answer.length(); ++i)
        answer[i] = std::tolower(answer[i],loc);
    return answer;
}

std::string str_to_lower(char *answer) {
    return str_to_lower(std::string(answer));
}

std::string abs_path(std::string path_name, std::string file_name) {
#ifdef _WIN32
    std::string delimiter = "\\";
#else
    std::string delimiter = "/";
#endif
    if (path_name.back()=='\\' || path_name.back()=='/') {
        return path_name + file_name;
    }
    return path_name + delimiter + file_name;
}

#ifdef OBJECT_DETECTION_ARMNN_TFLITE
float *readAnchorsFile(std::string filename, int& size) {
    std::vector<std::string> lines;
    lines.clear();
    std::ifstream file(filename);
    std::string s;
    while (getline(file, s))
        lines.push_back(s);

    size = lines.size();
    float *result = new float[size * 4]();
    for (int i = 0; i < size; i++) {
        int index = i * 4;
        std::stringstream(lines[i]) >> result[index]
                                    >> result[index + 1]
                                    >> result[index + 2]
                                    >> result[index + 3];
    }
    return result;
}
#endif //OBJECT_DETECTION_ARMNN_TFLITE

#endif //DETECT_SETTINGS_H
