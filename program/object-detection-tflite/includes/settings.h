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
#include <map>
#include <list>
#include <thread>

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
std::string abs_path(std::string, std::string);
std::string str_to_lower(std::string);
std::string str_to_lower(char *);
bool get_yes_no(std::string);
bool get_yes_no(char *);
std::vector<std::string> *readClassesFile(std::string);

class Settings {
public:
    Settings() {
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
        if (model_dataset_type != "coco") {
            throw ("Unsupported model dataset type: " + model_dataset_type);
        }

        std::string nms_type = alter_str(getenv("USE_NMS"), "regular");
        if (str_to_lower(nms_type) == "regular") {
            _graph_file = std::string(getenv("CK_ENV_TENSORFLOW_MODEL_TFLITE_GRAPH_REGULAR_NMS"));
            _fast_nms = false;
        } else {
            _graph_file = std::string(getenv("CK_ENV_TENSORFLOW_MODEL_TFLITE_GRAPH_FAST_NMS"));
            _fast_nms = true;
        }
        _graph_file = abs_path(std::string(getenv("CK_ENV_TENSORFLOW_MODEL_ROOT")), _graph_file);

        std::string classes_file = abs_path(std::string(getenv("CK_ENV_TENSORFLOW_MODEL_ROOT")),
                                            std::string(getenv("CK_ENV_TENSORFLOW_MODEL_CLASSES")));
        _model_classes = *readClassesFile(classes_file);
        _images_dir = settings_from_file["PREPROCESS_OUT_DIR"];
        _detections_out_dir = settings_from_file["DETECTIONS_OUT_DIR"];
        _images_file = settings_from_file["PREPROCESSED_FILES"];
        _image_size_height = std::stoi(settings_from_file["MODEL_IMAGE_HEIGHT"]);
        _image_size_width = std::stoi(settings_from_file["MODEL_IMAGE_WIDTH"]);
        _num_channels = std::stoi(settings_from_file["MODEL_IMAGE_CHANNELS"]);
        _correct_background = get_yes_no(settings_from_file["MODEL_NEED_BACKGROUND_CORRECTION"]);
        _normalize_img = get_yes_no(settings_from_file["MODEL_NORMALIZE_DATA"]);
        _subtract_mean = get_yes_no(settings_from_file["MODEL_SUBTRACT_MEAN"]);

        _use_neon = get_yes_no(getenv("USE_NEON"));
        _use_opencl = get_yes_no(getenv("USE_OPENCL"));
        _number_of_threads = std::thread::hardware_concurrency();
        _number_of_threads = _number_of_threads < 1 ? 1 : _number_of_threads;
        _number_of_threads = std::stoi(alter_str(getenv("CK_HOST_CPU_NUMBER_OF_PROCESSORS"), std::to_string(_number_of_threads)));
        _batch_count = std::stoi(alter_str(getenv("CK_BATCH_COUNT"), "1"));
        _batch_size = std::stoi(alter_str(getenv("CK_BATCH_SIZE"), "1"));
        _full_report = get_yes_no(getenv("FULL_REPORT"));
        _verbose = get_yes_no(getenv("VERBOSE"));

        _default_model_settings=!get_yes_no(getenv("USE_CUSTOM_NMS_SETTINGS"));

        if (_default_model_settings) {
            _m_max_classes_per_detection = 1;
            _m_max_detections = std::stoi(getenv("CK_ENV_TENSORFLOW_MODEL_MAX_DETECTIONS"));
            _m_detections_per_class = 100;
            _m_num_classes = std::stoi(getenv("CK_ENV_TENSORFLOW_MODEL_NUM_CLASSES"));
            _m_nms_score_threshold = std::stof(getenv("CK_ENV_TENSORFLOW_MODEL_NMS_SCORE_THRESHOLD"));
            _m_nms_iou_threshold = std::stof(getenv("CK_ENV_TENSORFLOW_MODEL_NMS_IOU_THRESHOLD"));
            _m_scale_h = std::stof(getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_H"));
            _m_scale_w = std::stof(getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_W"));
            _m_scale_x = std::stof(getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_X"));
            _m_scale_y = std::stof(getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_Y"));
        } else {
            _m_max_classes_per_detection = std::stoi(alter_str(getenv("MAX_CLASSES_PER_DETECTION"), "1"));
            if (_m_max_classes_per_detection > 1 && _fast_nms) {
                std::cout << std::endl << "You can't use USE_NMS=fast and MAX_CLASSES_PER_DETECTION>1 at the same time" << std::endl ;
                exit(-1);
            }

            _m_max_detections = std::stoi(alter_str(getenv("MAX_DETECTIONS"), getenv("CK_ENV_TENSORFLOW_MODEL_MAX_DETECTIONS")));
            _m_detections_per_class = std::stoi(alter_str(getenv("DETECTIONS_PER_CLASS"), "100"));
            _m_num_classes = std::stoi(alter_str(getenv("NUM_CLASSES"), getenv("CK_ENV_TENSORFLOW_MODEL_NUM_CLASSES")));
            _m_nms_score_threshold = std::stof(alter_str(getenv("NMS_SCORE_THRESHOLD"), getenv("CK_ENV_TENSORFLOW_MODEL_NMS_SCORE_THRESHOLD")));
            _m_nms_iou_threshold = std::stof(alter_str(getenv("NMS_IOU_THRESHOLD"), getenv("CK_ENV_TENSORFLOW_MODEL_NMS_IOU_THRESHOLD")));
            _m_scale_h = std::stof(alter_str(getenv("SCALE_H"), getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_H")));
            _m_scale_w = std::stof(alter_str(getenv("SCALE_W"), getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_W")));
            _m_scale_x = std::stof(alter_str(getenv("SCALE_X"), getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_X")));
            _m_scale_y = std::stof(alter_str(getenv("SCALE_Y"), getenv("CK_ENV_TENSORFLOW_MODEL_SCALE_Y")));
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
            std::cout << "Use NEON: " << _use_neon << std::endl;
            std::cout << "Use OPENCL: " << _use_opencl << std::endl;
        }

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

    bool use_neon() { return _use_neon; }

    bool use_opencl() { return _use_opencl; }

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
    bool _use_neon;
    bool _use_opencl;
    bool _verbose;
};

std::vector<std::string> *readClassesFile(std::string filename)
{
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

#endif //UNTITLED_SETTINGS_H
