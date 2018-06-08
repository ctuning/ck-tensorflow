#include <iostream>
#include <fstream>
#include <cstdlib>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/examples/label_image/get_top_n_impl.h"

#include "image_helper.h"

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
    // Parse command line arguments
    string image_file;
    string model_file;
    string labels_file;
    for (int i = 1; i < argc; i++)
      get_arg(argv[i], "--image=", image_file) ||
      get_arg(argv[i], "--graph=", model_file) ||
      get_arg(argv[i], "--labels=", labels_file);
    check_file(image_file, "Image");
    //check_file(model_file, "Model");
    //check_file(labels_file, "Labels");

    // Read input image
    ImageData img_data = load_jpeg_file(image_file);
    cout << "OK: Input image loaded: " << img_data.height << "x"
                                       << img_data.width << "x"
                                       << img_data.channels << endl;
    if (img_data.channels != 3)
      throw string("Only RGB images are supported");

    // Resize input image to model
    const int model_image_H = 224;
    const int model_image_W = 224;
    vector<float> img_data_resized = resize_image(img_data, model_image_H, model_image_W);
  }
  catch (const string& error_message) {
    cout << "ERROR: " << error_message << endl;
    return -1;
  }

  /*tflite::label_image::Settings s;
  s.model_name = getenv("RUN_OPT_MODEL");
  s.input_bmp_name = getenv("RUN_OPT_IMAGE");
  s.labels_file_name = getenv("RUN_OPT_LABELS");

  // Load network from ftlite file
  unique_ptr<tflite::FlatBufferModel> model;
  model = tflite::FlatBufferModel::BuildFromFile(s.model_name.c_str());
  if (!model) {
    cerr << "ERROR: Failed to load model " << s.model_name << endl;
    return -1;
  }
  cout << "OK: Model loaded: " << s.model_name << endl;
*/
  // Build interpter
/*  unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    cerr << "ERROR: Failed to construct interpreter" << endl;
    return -1;
  }
  cout << "OK: Interpreter constructed" << endl;
  interpreter->UseNNAPI(s.accel);
  interpreter->SetNumThreads(s.number_of_threads);

  // Load test input image
  int image_width = 224;
  int image_height = 224;
  int image_channels = 3;
  uint8_t* input_data = tflite::label_image::read_bmp(
    s.input_bmp_name.c_str(), &image_width, &image_height, &image_channels, &s);
  cout << "OK: Input image loaded: " << s.input_bmp_name << endl;
  int input = interpreter->inputs()[0];
  int output = interpreter->outputs()[0];
  if (interpreter->tensor(input)->type != kTfLiteUInt8 ||
      interpreter->tensor(output)->type != kTfLiteUInt8) {
    cerr << "ERROR: this demo is for UINT8 input/output only" << endl;
    return -1;
  }

  // Allocate memory
  const vector<int> inputs = interpreter->inputs();
  const vector<int> outputs = interpreter->outputs();
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    cerr << "ERROR: Failed to allocate tensors" << endl;
    return -1;
  }

  // Prepare input image
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];
  tflite::label_image::resize<uint8_t>(
    interpreter->typed_tensor<uint8_t>(input), input_data,
    image_height, image_width, image_channels,
    wanted_height, wanted_width, wanted_channels, &s);
  cout << "OK: Image resized" << endl;

  // Classify image
  if (interpreter->Invoke() != kTfLiteOk) {
    cerr << "ERROR: Failed to invoke tflite" << endl;
    return -1;
  }
  cout << "OK: Image classified" << endl;

  // Process results
  const int output_size = 1000;
  const size_t num_results = 5;
  const float threshold = 0.001f;
  vector<pair<float, int>> top_results;
  tflite::label_image::get_top_n<uint8_t>(
    interpreter->typed_output_tensor<uint8_t>(0),
    output_size, num_results, threshold, &top_results, false);

  // Read labels
  vector<string> labels;
  ifstream file(s.labels_file_name);
  string line;
  while (getline(file, line))
    labels.push_back(line);

  // Print predictions
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    cout << confidence << ": " << index << " " << labels[index] << endl;
  }
*/
  return 0;
}
