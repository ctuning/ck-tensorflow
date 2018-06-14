/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#include "benchmark.h"

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"

using namespace std;
using namespace CK;

template <typename TData, typename TInConverter, typename TOutConverter>
class Benchmark {
public:
  Benchmark(const BenchmarkSettings* settings, const tflite::Interpreter* interpreter, int input_index) {
    _in_ptr = interpreter->typed_tensor<TData>(input_index);
    _out_ptr = interpreter->typed_output_tensor<TData>(0);
    _in_data = new ImageData(settings);
    _out_data = new ResultData(settings);
    _in_converter = new TInConverter(settings);
    _out_converter = new TOutConverter(settings);
  }

  void load_images(const vector<string>& batch_images) {
    int offset = 0;
    for (auto image_file : batch_images) {
      _in_data->load(image_file);
      _in_converter->convert(_in_data, _in_ptr + offset);
      offset += _in_data.length();
    }
  }

  void save_results(const vector<string>& batch_images) {
    int offset = 0;
    for (auto image_file : batch_images) {
      _out_converter->convert(_out_ptr + offset, _out_data);
      _out_data->save(image_file);
      offset += _out_data.length();
    }
  }

private:
  TData* _in_ptr;
  TData* _out_ptr;
  unique_ptr<ImageData> _in_data;
  unique_ptr<ResultData> _out_data;
  unique_ptr<TInConverter> _in_converter;
  unique_ptr<TOutConverter> _out_converter;
};


int main(int argc, char* argv[]) {
  try {
    init_benchmark();
    
    BenchmarkSettings settings;
    BenchmarkSession session(&settings);

    unique_ptr<Benchmark> benchmark;
    unique_ptr<tflite::FlatBufferModel> model;
    unique_ptr<tflite::Interpreter> interpreter;

    cout << "Loading graph..." << endl;
    measure_setup([&]{
      model = tflite::FlatBufferModel::BuildFromFile(graph_file.c_str());
      if (!model)
        throw "Failed to load graph from file " + graph_file;

      tflite::ops::builtin::BuiltinOpResolver resolver;
      tflite::InterpreterBuilder(*model, resolver)(&interpreter);
      if (!interpreter)
        throw string("Failed to construct interpreter");
      if (interpreter->AllocateTensors() != kTfLiteOk)
        throw string("Failed to allocate tensors");
        
      int input_index = interpreter->inputs()[0];
      int output_index = interpreter->outputs()[0];
      auto input_type = interpreter->tensor(input_index)->type;
      auto output_type = interpreter->tensor(output_index)->type;

      if (input_type != output_type)
        throw format("Type of graph's input (%d) does not match type of its output (%d).",
                     int(input_type), int(output_type));

      switch (input_type) {
      case kTfLiteFloat32:
        benchmark = new Benchmark<float, InNormalize, OutCopy>(&settings, interpreter, input_index);
        break;

      case kTfLiteUInt8:
        benchmark = new Benchmark<uint8_t, InCopy, OutDequantize>(&settings, interpreter, input_index);
        break;

      default:
        throw format("Unsupported type of graph's input: %d. "
                     "Supported types are: Float32 (%d), UInt8 (%d)",
                     int(input_type), int(kTfLiteFloat32), int(kTfLiteUInt8));
      }

      TfLiteIntArray* dims = interpreter->tensor(input_index)->dims;
      int wanted_height = dims->data[1];
      int wanted_width = dims->data[2];
      int wanted_channels = dims->data[3];
      if (wanted_height != settings.image_size ||
          wanted_width != settings.image_size ||
          wanted_channels != settings.NUM_CHANNELS)
        throw format("Dimensions of graph's input (HWC = %dx%dx%d) "
                     "do not correspond to dimensions of input image (%dx%dx%d)",
                     wanted_height, wanted_width, wanted_channels,
                     settings.image_size, settings.image_size, settings.NUM_CHANNELS);
    });

    cout << "Processing batches...";
    measure_prediction([&]{
      while (session->get_next_batch()) {
        cout << "\nBatch " << session.batch_index()+1 << " of " << settings.batch_count << endl;
    
        session->measure_begin();
        benchmark->load_images(session->batch_files());
        session->measure_end_load_images();

        session->measure_begin();
        if (interpreter->Invoke() != kTfLiteOk)
          throw "Failed to invoke tflite"
        session->measure_end_prediction();

        benchmark->save_results(session->batch_files());
      }
    });

    finish_benchmark(session);
  }
  catch (const string& error_message) {
    cerr << "ERROR: " << error_message << endl;
    return -1;
  }
  return 0;
}
