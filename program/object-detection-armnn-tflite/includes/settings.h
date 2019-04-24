//
// Created by ivan on 4/23/19.
//

#ifndef UNTITLED_SETTINGS_H
#define UNTITLED_SETTINGS_H

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <dirent.h>


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

bool get_yes_no(std::string answer) {
    std::locale loc;
    for (std::string::size_type i=0; i<answer.length(); ++i)
        answer[i] = std::tolower(answer[i],loc);
    if (answer == "1" || answer == "yes") return true;
    return false;
}
bool get_yes_no(char *answer) {
    if (answer == nullptr) return false;
    return get_yes_no(std::string(answer));
}

struct FileInfo {
    std::string name;
    int width;
    int height;
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

        _graph_file = std::string(getenv("CK_ENV_TENSORFLOW_MODEL_ROOT")) + "/" + getenv("CK_ENV_TENSORFLOW_MODEL_TFLITE_GRAPH_NO_POSTPROCESSING");

        std::string classes_file = std::string(getenv("CK_ENV_TENSORFLOW_MODEL_ROOT")) + "/" +
                                   getenv("CK_ENV_TENSORFLOW_MODEL_CLASSES");
        _model_classes = *readClassesFile(classes_file);

        std::string anchors_file = std::string(getenv("CK_ENV_TENSORFLOW_MODEL_ROOT")) + "/" +
                                   getenv("CK_ENV_TENSORFLOW_MODEL_ANCHORS");
        _m_anchors = readAnchorsFile(anchors_file, _m_anchors_count);

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

        _number_of_threads = std::stoi(alter_str(getenv("CK_HOST_CPU_NUMBER_OF_PROCESSORS"), "1"));
        _batch_count = std::stoi(alter_str(getenv("CK_BATCH_COUNT"), "1"));
        _batch_size = std::stoi(alter_str(getenv("CK_BATCH_SIZE"), "1"));
        _full_report = get_yes_no(getenv("FULL_REPORT"));
        _verbose = get_yes_no(getenv("VERBOSE"));

        _m_anchors_count = 1917;
        _m_max_classes_per_detection = std::stoi(alter_str(getenv("MAX_CLASSES_PER_DETECTION"), "100"));
        _m_max_detections = std::stoi(alter_str(getenv("MAX_DETECTIONS"), "100"));
        _m_detections_per_class = std::stoi(alter_str(getenv("DETECTIONS_PER_CLASS"), "100"));
        _m_num_classes = std::stoi(alter_str(getenv("NUM_CLASSES"), "91"));
        _m_nms_score_threshold = std::stof(alter_str(getenv("NMS_SCORE_THRESHOLD"), "0.3"));
        _m_nms_iou_threshold = std::stof(alter_str(getenv("NMS_IOU_THRESHOLD"), "0.6"));
        _m_h_scale = std::stof(alter_str(getenv("H_SCALE"), "5.0"));
        _m_w_scale = std::stof(alter_str(getenv("W_SCALE"), "5.0"));
        _m_x_scale = std::stof(alter_str(getenv("X_SCALE"), "10.0"));
        _m_y_scale = std::stof(alter_str(getenv("Y_SCALE"), "10.0"));

        _d_boxes = new float [_m_anchors_count * 4]();
        _d_scores = new float [_m_anchors_count * _m_num_classes];

        _d_scores_sort_buf = new float[_m_max_detections + _m_max_classes_per_detection];
        _d_classes_sort_buf = new int[_m_max_detections + _m_max_classes_per_detection];
        _d_boxes_sort_buf = new int[_m_max_detections + _m_max_classes_per_detection];
        _d_classes_ids_sort_buf = new int[_m_num_classes];

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

    ~Settings() {
        delete _d_boxes;
        delete _d_scores;
        delete _d_scores_sort_buf;
        delete _d_classes_sort_buf;
        delete _d_boxes_sort_buf;
        delete _d_classes_ids_sort_buf;
    }
    const std::vector<FileInfo> &image_list() const { return _image_list; }

    const std::vector<std::string> &model_classes() const { return _model_classes; }

    int batch_count() { return _batch_count; }

    int batch_size() { return _batch_size; }

    int detections_buffer_size() { return _m_max_detections + 1; }

    int get_anchors_count() { return _m_anchors_count; }

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

    float get_h_scale() { return _m_h_scale; };
    void set_h_scale(float i) { _m_h_scale = i;}

    float get_w_scale() { return _m_w_scale; };
    void set_w_scale(float i) { _m_w_scale = i;}

    float get_x_scale() { return _m_x_scale; };
    void set_x_scale(float i) { _m_x_scale = i;}

    float get_y_scale() { return _m_y_scale; };
    void set_y_scale(float i) { _m_y_scale = i;}

    float* get_anchors() { return _m_anchors; };
    float* get_boxes_buf() { return _d_boxes; };
    float* get_scores_buf() { return _d_scores; };
    float* get_scores_sorting_buf() { return _d_scores_sort_buf; };

    int* get_classes_sorting_buf() { return _d_classes_sort_buf; }
    int* get_classes_ids_sorting_buf() { return _d_classes_ids_sort_buf; }
    int* get_boxes_sorting_buf() { return _d_boxes_sort_buf; }

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

    int *_d_classes_sort_buf;
    int *_d_boxes_sort_buf;
    int *_d_classes_ids_sort_buf;
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
    int _m_anchors_count;

    float *_d_boxes;
    float *_d_scores;
    float *_d_scores_sort_buf;
    float *_m_anchors;

    float _m_nms_score_threshold;
    float _m_nms_iou_threshold;
    float _m_h_scale;
    float _m_w_scale;
    float _m_x_scale;
    float _m_y_scale;

    bool _correct_background;
    bool _full_report;
    bool _normalize_img;
    bool _subtract_mean;
    bool _use_neon;
    bool _use_opencl;
    bool _verbose;
};

#endif //UNTITLED_SETTINGS_H
