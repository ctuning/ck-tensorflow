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


class IBenchmark {
public:
  bool has_background_class = false;

  virtual void load_images(const vector<string>& batch_images) = 0;
  virtual void save_results(const vector<string>& batch_images) = 0;
};


template <typename TData, typename TInConverter, typename TOutConverter>
class Benchmark : public IBenchmark {
public:
  Benchmark(const BenchmarkSettings* settings, tflite::Interpreter* interpreter, int input_index) {
    _in_ptr = interpreter->typed_tensor<TData>(input_index);
    _out_ptr = interpreter->typed_output_tensor<TData>(0);
    _in_data.reset(new ImageData(settings));
    _out_data.reset(new ResultData(settings));
    _in_converter.reset(new TInConverter(settings));
    _out_converter.reset(new TOutConverter(settings));
  }

  void load_images(const vector<string>& batch_images) override {
    int image_offset = 0;
    for (auto image_file : batch_images) {
      _in_data->load(image_file);
      _in_converter->convert(_in_data.get(), _in_ptr + image_offset);
      image_offset += _in_data->size();
    }
  }

  void save_results(const vector<string>& batch_images) override {
    int image_offset = 0;
    int probe_offset = has_background_class ? 1 : 0;
    for (auto image_file : batch_images) {
      _out_converter->convert(_out_ptr + image_offset + probe_offset, _out_data.get());
      _out_data->save(image_file);
      image_offset += _out_data->size() + probe_offset;
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

    // TODO: learn how to process batches via tflite.
    // currently interpreter->tensor(input_index)->dims[0] = 1
    if (settings.batch_size != 1)
      throw string("Only BATCH_SIZE=1 is currently supported");
    
    BenchmarkSession session(&settings);

    unique_ptr<IBenchmark> benchmark;
    unique_ptr<tflite::FlatBufferModel> model;
    unique_ptr<tflite::Interpreter> interpreter;

    cout << "\nLoading graph..." << endl;
    measure_setup([&]{
      model = tflite::FlatBufferModel::BuildFromFile(settings.graph_file.c_str());
      if (!model)
        throw "Failed to load graph from file " + settings.graph_file;

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
        benchmark.reset(new Benchmark<float, InNormalize, OutCopy>(&settings, interpreter.get(), input_index));
        break;

      case kTfLiteUInt8:
        benchmark.reset(new Benchmark<uint8_t, InCopy, OutDequantize>(&settings, interpreter.get(), input_index));
        break;

      default:
        throw format("Unsupported type of graph's input: %d. "
                     "Supported types are: Float32 (%d), UInt8 (%d)",
                     int(input_type), int(kTfLiteFloat32), int(kTfLiteUInt8));
      }

      TfLiteIntArray* in_dims = interpreter->tensor(input_index)->dims;
      int in_num = in_dims->data[0];
      int in_height = in_dims->data[1];
      int in_width = in_dims->data[2];
      int in_channels = in_dims->data[3];
      cout << format("Input tensor dimensions (NHWC): %d*%d*%d*%d", in_num, in_height, in_width, in_channels) << endl;
      if (in_height != settings.image_size ||
          in_width != settings.image_size ||
          in_channels != settings.num_channels)
        throw format("Dimensions of graph's input do not correspond to dimensions of input image (%d*%d*%d*%d)",
                     settings.batch_size, settings.image_size, settings.image_size, settings.num_channels);

      TfLiteIntArray* out_dims = interpreter->tensor(output_index)->dims;
      int out_num = out_dims->data[0];
      int out_classes = out_dims->data[1];
      cout << format("Output tensor dimensions: %d*%d", out_num, out_classes) << endl;
      if (out_classes != settings.num_classes && out_classes != settings.num_classes+1)
        throw format("Unsupported number of classes in graph's output tensor. Supported numbers are %d and %d",
                     settings.num_classes, settings.num_classes+1);
      benchmark->has_background_class = out_classes == settings.num_classes+1;
    });

    cout << "\nProcessing batches..." << endl;
    measure_prediction([&]{
      while (session.get_next_batch()) {
        session.measure_begin();
        benchmark->load_images(session.batch_files());
        session.measure_end_load_images();

        session.measure_begin();
        if (interpreter->Invoke() != kTfLiteOk)
          throw "Failed to invoke tflite";
        session.measure_end_prediction();

        benchmark->save_results(session.batch_files());
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
