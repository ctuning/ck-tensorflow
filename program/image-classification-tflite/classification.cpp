/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#include "benchmark.h"

#ifdef TF_LITE_1_13
  #include "tensorflow/lite/kernels/register.h"
  #include "tensorflow/lite/model.h"
#else
  #include "tensorflow/contrib/lite/kernels/register.h"
  #include "tensorflow/contrib/lite/model.h"
#endif

using namespace std;
using namespace CK;


template <typename TData, typename TInConverter, typename TOutConverter>
class TFLiteBenchmark : public Benchmark<TData, TInConverter, TOutConverter> {
public:
  TFLiteBenchmark(const BenchmarkSettings* settings, tflite::Interpreter* interpreter, int input_index)
    : Benchmark<TData, TInConverter, TOutConverter>(
      settings, interpreter->typed_tensor<TData>(input_index), interpreter->typed_output_tensor<TData>(0)) {
  }
};


int main(int argc, char* argv[]) {
  try {
    init_benchmark();
    
    BenchmarkSettings settings(MODEL_TYPE::LITE);

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
      model = tflite::FlatBufferModel::BuildFromFile(settings.graph_file().c_str());
      if (!model)
        throw "Failed to load graph from file " + settings.graph_file();

      tflite::ops::builtin::BuiltinOpResolver resolver;
      tflite::InterpreterBuilder(*model, resolver)(&interpreter);
      if (!interpreter)
        throw string("Failed to construct interpreter");
      if (interpreter->AllocateTensors() != kTfLiteOk)
        throw string("Failed to allocate tensors");

      interpreter->SetNumThreads(settings.number_of_threads());

      int input_index = interpreter->inputs()[0];
      int output_index = interpreter->outputs()[0];
      auto input_type = interpreter->tensor(input_index)->type;
      auto output_type = interpreter->tensor(output_index)->type;
      if (input_type != output_type)
        throw format("Type of graph's input (%d) does not match type of its output (%d).",
                     int(input_type), int(output_type));

      switch (input_type) {
        case kTfLiteFloat32:
          if (settings.skip_internal_preprocessing)
            benchmark.reset(new TFLiteBenchmark<float, InCopy, OutCopy>(&settings, interpreter.get(), input_index));
          else
            benchmark.reset(new TFLiteBenchmark<float, InNormalize, OutCopy>(&settings, interpreter.get(), input_index));
          break;

        case kTfLiteUInt8:
          benchmark.reset(new TFLiteBenchmark<uint8_t, InCopy, OutDequantize>(&settings, interpreter.get(), input_index));
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
