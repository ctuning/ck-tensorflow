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

#include <xopenme.h>

using namespace std;
using namespace tensorflow;

inline int getenv_i(const char *name, int def) {
  return getenv(name) ? atoi(getenv(name)) : def;
}

enum GLOBAL_TIMER {
  X_TIMER_SETUP,
  X_TIMER_TEST,

  GLOBAL_TIMER_COUNT
};

enum GLOBAL_VAR {
  VAR_TIME_SETUP,
  VAR_TIME_TEST,
  VAR_TIME_IMG_LOAD_TOTAL,
  VAR_TIME_IMG_LOAD_AVG,
  VAR_TIME_CLASSIFY_TOTAL,
  VAR_TIME_CLASSIFY_AVG,

  GLOBAL_VAR_COUNT
};

inline void store_value_f(int index, const char* name, float value) {
  char* json_name = new char[strlen(name) + 6];
  sprintf(json_name, "\"%s\":%%f", name);
  xopenme_add_var_f(index, json_name, value);
  delete[] json_name;
}

const int NUM_CHANNELS = 3;
const int NUM_CLASSES = 1001;

int main(int argc, char* argv[]) {
  xopenme_init(GLOBAL_TIMER_COUNT, GLOBAL_VAR_COUNT);

  // Load parameters
  string graph_file(getenv("RUN_OPT_FROZEN_GRAPH"));
  string images_dir(getenv("RUN_OPT_IMAGE_DIR"));
  string images_file(getenv("RUN_OPT_IMAGE_LIST"));
  string result_dir(getenv("RUN_OPT_RESULT_DIR"));
  string input_layer_name(getenv("RUN_OPT_INPUT_LAYER_NAME"));
  string output_layer_name(getenv("RUN_OPT_OUTPUT_LAYER_NAME"));
  int batch_count = getenv_i("RUN_OPT_BATCH_COUNT", 1);
  int batch_size = getenv_i("RUN_OPT_BATCH_SIZE", 1);
  int image_size = getenv_i("RUN_OPT_IMAGE_SIZE", 224);
  bool normalize_img = getenv_i("RUN_OPT_NORMALIZE_DATA", 1) == 1;
  bool subtract_mean = getenv_i("RUN_OPT_SUBTRACT_MEAN", 1) == 1;

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
  
  //-------------------------------------------------
  // Load frozen graph
  cout << "Loading frozen graph..." << endl;
  xopenme_clock_start(X_TIMER_SETUP);
  unique_ptr<Session> session;
  GraphDef graph_def;
  Status load_graph_status = ReadBinaryProto(Env::Default(), graph_file, &graph_def);
  if (!load_graph_status.ok()) {
    cerr << "Failed to load graph: " << load_graph_status.ToString() << endl;
    return -1;
  }
  session.reset(NewSession(SessionOptions()));
  Status session_create_status = session->Create(graph_def);
  if (!session_create_status.ok()) {
    cerr << "Failed to create new session: " << session_create_status.ToString() << endl;
    return -1;
  }
  xopenme_clock_end(X_TIMER_SETUP);

  //-------------------------------------------------
  // Classify each batch
  xopenme_clock_start(X_TIMER_TEST);
  int img_index = 0;
  int img_px_count = image_size * image_size * NUM_CHANNELS;
  double total_load_images_time = 0;
  double total_prediction_time = 0;
  vector<uint8_t> img_data(img_px_count);
  Tensor input(DT_FLOAT, TensorShape({batch_size, image_size, image_size, NUM_CHANNELS}));
  float* input_ptr = input.flat<float>().data();
  for (int batch_index = 0; batch_index < batch_count; batch_index++) {
    cout << "\nProcessing batch " << batch_index << " of " << batch_count << "...\n";

    //-------------------------------------------------
    // Load batch of images into input tensor
    auto load_start_time = chrono::high_resolution_clock::now();
    for (int batch_img_index = 0; batch_img_index < batch_size; batch_img_index++) {
      // Read image data bytes
      auto image_path = images_dir + '/' + image_list[img_index];
      ifstream file(image_path, ios::in | ios::binary);
      if (!file.good()) {
        cerr << "Failed to open image data " + image_path << endl;
        return -1;
      }
      file.read(reinterpret_cast<char*>(img_data.data()), img_px_count);

      // Copy image data to input tensor
      float sum = 0;
      int img_offset = batch_img_index * img_px_count;
      for (int px_offset = 0; px_offset < img_px_count; px_offset++) {
        //float px = img_buf[px_offset];
        float px = img_data[px_offset];
        if (normalize_img)
          px = (px / 255.0 - 0.5) * 2.0;
        sum += px;
        input_ptr[img_offset + px_offset] = px;
      }
      
      // Subtract mean value if required
      if (subtract_mean) {
        float mean = sum / static_cast<float>(img_px_count);
        for (int px_offset = 0; px_offset < img_px_count; px_offset++)
          input_ptr[img_offset + px_offset] -= mean;
      }
      img_index++;
    }
    auto load_finish_time = chrono::high_resolution_clock::now(); 
    chrono::duration<double> load_time = load_finish_time - load_start_time;
    cout << "Batch loaded in " << load_time.count() << " s" << endl;
    total_load_images_time += load_time.count();

    //-------------------------------------------------
    // Classify current batch
    auto pred_start_time = chrono::high_resolution_clock::now();
    vector<Tensor> outputs;
    Status run_status = session->Run(
      {{input_layer_name, input}}, {output_layer_name}, {}, &outputs);
    if (!run_status.ok()) {
      cerr << "Running model failed: " << run_status.ToString() << endl;
      return -1;
    }
    auto pred_finish_time = chrono::high_resolution_clock::now(); 
    chrono::duration<double> pred_time = pred_finish_time - pred_start_time;
    cout << "Batch classified in " << pred_time.count() << " s" << endl;
    total_prediction_time += pred_time.count();

    //-------------------------------------------------
    // Process output tensor
    auto output_flat = outputs[0].flat<float>();
    if (output_flat.size() != batch_size * NUM_CLASSES) {
      cerr << "Invalid output tensor size " << output_flat.size()
           << "but expected size is " << batch_size * NUM_CLASSES << endl;
      return -1;
    }
    for (int batch_img_index = 0; batch_img_index < batch_size; batch_img_index++) {
      int img_index = batch_index * batch_size + batch_img_index;
      auto result_path = result_dir + '/' + image_list[img_index] + ".txt";
      ofstream file(result_path);
      if (!file.good()) {
        cerr << "Unable to create result file " + result_path << endl;
        return -1;
      }
      int probe_offset = batch_img_index * NUM_CLASSES;
      for (int class_index = 0; class_index < NUM_CLASSES; class_index++) {
        float probe_for_class = output_flat(probe_offset + class_index);
        file << probe_for_class << endl;
      }
    }
  }
  xopenme_clock_end(X_TIMER_TEST);

  //-------------------------------------------------
  // Store some metrics
  float setup_time = xopenme_get_timer(X_TIMER_SETUP);
  float test_time = xopenme_get_timer(X_TIMER_TEST);
  float img_count_f = static_cast<float>(image_list.size());
  float avg_load_images_time = total_load_images_time / img_count_f;
  float avg_prediction_time = total_prediction_time / img_count_f;

  cout << "-------------------------------\n";
  cout << "Graph loaded in " << setup_time << " s" << endl;
  cout << "All batches loaded in " << total_load_images_time << " s" << endl;
  cout << "All batches classified in " << total_prediction_time << " s" << endl;
  cout << "Average classification time: " << avg_prediction_time << " s" << endl;
  cout << "-------------------------------\n";

  store_value_f(VAR_TIME_SETUP, "setup_time_s", setup_time);
  store_value_f(VAR_TIME_TEST, "test_time_s", test_time);
  store_value_f(VAR_TIME_IMG_LOAD_TOTAL, "images_load_time_s", total_load_images_time);
  store_value_f(VAR_TIME_IMG_LOAD_AVG, "images_load_time_avg_s", avg_load_images_time);
  store_value_f(VAR_TIME_CLASSIFY_TOTAL, "prediction_time_total_s", total_prediction_time);
  store_value_f(VAR_TIME_CLASSIFY_AVG, "prediction_time_avg_s", avg_prediction_time);
  xopenme_dump_state();
  xopenme_finish();

  return 0;
}
