/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#include <fstream>
#include <chrono>

#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/framework/scope.h"

using namespace std;
using namespace tensorflow;

inline int getenv_i(const char *name, int def) {
  return getenv(name) ? atoi(getenv(name)) : def;
}

const int num_channels = 3;
const int num_classes = 1001;

int main(int argc, char* argv[]) {
  // Load parameters
  string graph_file(getenv("RUN_OPT_FROZEN_GRAPH"));
  string images_dir(getenv("RUN_OPT_IMAGE_DIR"));
  string images_file(getenv("RUN_OPT_IMAGE_LIST"));
  string result_dir(getenv("RUN_OPT_RESULT_DIR"));
  string input_layer_name("input"); // TODO extract
  string output_layer_name("MobilenetV1/Predictions/Reshape_1"); // TODO extract
  int batch_count = getenv_i("RUN_OPT_BATCH_COUNT", 1);
  int batch_size = getenv_i("RUN_OPT_BATCH_SIZE", 1);
  int image_size = getenv_i("RUN_OPT_IMAGE_SIZE", 224);
  bool normalize_img = getenv_i("RUN_OPT_NORMALIZE_DATA", 1) == 1;

  cout << "Graph file: " << graph_file << endl;
  cout << "Image dir: " << images_dir << endl;
  cout << "Image list: " << images_file << endl;
  cout << "Image size: " << image_size << endl;
  cout << "Result dir: " << result_dir << endl;
  cout << "Batch count: " << batch_count << endl;
  cout << "Batch size: " << batch_size << endl;

  // Load image filenames
  vector<string> image_list;
  ifstream file(images_file);
  if (!file.good()) {
    cerr << "Unable to open image list file " << images_file;
    return -1;
  }
  for (string s; !getline(file, s).fail();)
    image_list.emplace_back(s);
  cout << "Image count in file: " << image_list.size() << endl;
  
  // Load frozen graph
  cout << "Loading frozen graph..." << endl;
  auto start_time = chrono::high_resolution_clock::now();  
  unique_ptr<Session> session;
  GraphDef graph_def;
  Status load_graph_status = ReadBinaryProto(Env::Default(), graph_file, &graph_def);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << "Failed to load graph: " << load_graph_status;
    return -1;
  }
  session.reset(NewSession(SessionOptions()));
  Status session_create_status = session->Create(graph_def);
  if (!session_create_status.ok()) {
    LOG(ERROR) << "Failed to create new session: " << session_create_status;
    return -1;
  }
  auto finish_time = chrono::high_resolution_clock::now(); 
  chrono::duration<double> elapsed = finish_time - start_time;
  cout << "Graph loaded in " << elapsed.count() << "s" << endl;

  // Classify each batch
  int img_index = 0;
  int img_size_bytes = image_size * image_size * num_channels;
  uint8_t *img_buf = new uint8_t[img_size_bytes];
  Tensor input(DT_FLOAT, TensorShape({batch_size, image_size, image_size, num_channels}));
  float* input_ptr = input.flat<float>().data();
  try {
    for (int batch_index = 0; batch_index < batch_count; batch_index++) {
      cout << "Processing batch " << batch_index << " of " << batch_count << "...\n";

      // Load batch of images into input tensor
      for (int batch_img_index = 0; batch_img_index < batch_size; batch_img_index++) {
        auto image_path = images_dir + '/' + image_list[img_index];
        ifstream file(image_path, ios::in | ios::binary);
        if (!file.good())
          throw "Failed to open image data " + image_path;
        file.read(reinterpret_cast<char*>(img_buf), img_size_bytes);
        // Copy image data to input tensor
        float sum = 0;
        int img_offset = batch_img_index * img_size_bytes;
        for (int px_offset = 0; px_offset < img_size_bytes; px_offset++) {
          float px = img_buf[px_offset];
          if (normalize_img)
            px = (px / 255.0 - 0.5) * 2.0;
          sum += px;
          input_ptr[img_offset + px_offset] = px;
        }
        cout << endl;
        sum /= static_cast<float>(img_size_bytes);
        cout << "MEAN " << sum << endl;
        img_index++;
      }

      // Classify current batch
      vector<Tensor> outputs;
      Status run_status = session->Run(
        {{input_layer_name, input}}, {output_layer_name}, {}, &outputs);
      if (!run_status.ok())
        throw "Running model failed: " + run_status.ToString();

      // Process output tensor
      auto output = outputs[0];
      if (output.shape().dim_size(0) != batch_size)
        throw "Invalid number of outputs";
      if (output.shape().dim_size(1) != num_classes)
        throw "Invalid output height";
      auto output_flat = output.flat<float>();
      for (int batch_img_index = 0; batch_img_index < batch_size; batch_img_index++) {
        int img_index = batch_index * batch_size + batch_img_index;
        auto result_path = result_dir + '/' + image_list[img_index] + ".txt";
        ofstream file(result_path);
        if (!file.good())
          throw "Unable to create result file " + result_path;
        for (int class_index = 0; class_index < num_classes; class_index++)
          file << output_flat(batch_img_index*num_classes + class_index) << endl;
      }
    }
    delete[] img_buf;
  }
  catch (const string& error_msg) {
    cerr << "ERROR: " << error_msg << endl;
    delete[] img_buf;
    return -1;
  }
  return 0;
}
