#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/examples/label_image/get_top_n_impl.h"

#ifdef XOPENME
#include <xopenme.h>
#endif

#include "image_helper.h"

enum X_TIMERS {
  X_TIMER_SETUP,
  X_TIMER_LOAD_IMAGE,
  X_TIMER_CLASSIFY,

  X_TIMER_COUNT
}; 

using namespace std;

bool get_arg(const char* arg, const char* key, string& target) {
  if (strncmp(arg, key, strlen(key)) == 0) {
    target = &arg[strlen(key)];
    return true;
  }
  return false;
}

void check_file(const string& path, const string& id) {
  if (path.empty())
    throw id + " file path is not specified";
  if (!ifstream(path).good())
    throw id + " file can't be opened, check if it exists";
  cout << id << " file: " << path << endl;
}

int main(int argc, char *argv[]) {
  try {
#ifdef XOPENME
    xopenme_init(X_TIMER_COUNT, 0);
#endif

    // Parse command line arguments
    string image_file;
    string model_file;
    string labels_file;
    for (int i = 1; i < argc; i++) {
      get_arg(argv[i], "--image=", image_file) ||
      get_arg(argv[i], "--graph=", model_file) ||
      get_arg(argv[i], "--labels=", labels_file);
    }
    check_file(image_file, "Image");
    check_file(model_file, "Model");
    check_file(labels_file, "Labels");

    // Load network from ftlite file
#ifdef XOPENME
    xopenme_clock_start(X_TIMER_SETUP);
#endif
    unique_ptr<tflite::FlatBufferModel> model;
    model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    if (!model)
      throw "Failed to load model " + model_file;

    unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter)
      throw string("Failed to construct interpreter");
    if (interpreter->AllocateTensors() != kTfLiteOk)
      throw string("Failed to allocate tensors");
      
    int input = interpreter->inputs()[0];
    int output = interpreter->outputs()[0];
    if (interpreter->tensor(input)->type != kTfLiteFloat32 ||
        interpreter->tensor(output)->type != kTfLiteFloat32)
      throw string("This demo is for FLOAT input/output only");
#ifdef XOPENME
    xopenme_clock_end(X_TIMER_SETUP);
#endif

    // Read input image
#ifdef XOPENME
    xopenme_clock_start(X_TIMER_LOAD_IMAGE);
#endif
    ImageData img_data = load_jpeg_file(image_file);
    cout << "OK: Input image loaded: " << img_data.height << "x"
                                       << img_data.width << "x"
                                       << img_data.channels << endl;
    if (img_data.channels != 3)
      throw string("Only RGB images are supported");

    // Prepare input image
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_channels = dims->data[3];
    if (wanted_channels != img_data.channels)
      throw string("Unsupported channels number in model");
    resize_image(interpreter->typed_tensor<float>(input), img_data, wanted_height, wanted_width);
#ifdef XOPENME
    xopenme_clock_end(X_TIMER_LOAD_IMAGE);
#endif

    // Classify image
    long ct_repeat_max = getenv("CT_REPEAT_MAIN") ? atol(getenv("CT_REPEAT_MAIN")) : 1;
#ifdef XOPENME
    xopenme_clock_start(X_TIMER_CLASSIFY);
#endif
    for (int i = 0; i < ct_repeat_max; i++) {
      if (interpreter->Invoke() != kTfLiteOk)
        throw string("Failed to invoke tflite");
    }
#ifdef XOPENME
    xopenme_clock_end(X_TIMER_CLASSIFY);
#endif
    cout << "OK: Image classified" << endl;

    // Process results
    const int output_size = 1000;
    const size_t num_results = 5;
    const float threshold = 0.0001f;
    vector<pair<float, int>> top_results;
    tflite::label_image::get_top_n<float>(
      interpreter->typed_output_tensor<float>(0),
      output_size, num_results, threshold, &top_results, true);

    // Read labels
    vector<string> labels;
    ifstream file(labels_file);
    string line;
    while (getline(file, line))
      labels.push_back(line);

    // Print predictions
    cout << "---------- Prediction for " << image_file << " ----------" << endl;
    for (const auto& result : top_results) {
      const float confidence = result.first;
      const int index = result.second;
      cout << fixed << setprecision(4) << confidence 
        << " - \"" << labels[index] << " (" << index << ")\"" << endl;
    }

#ifdef XOPENME
    xopenme_dump_state();
    xopenme_finish();
#endif    
  }
  catch (const string& error_message) {
    cout << "ERROR: " << error_message << endl;
    return -1;
  }
  return 0;
}
