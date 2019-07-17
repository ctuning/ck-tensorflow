/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#include <future>

#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"


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

class Program {
public:
  Program () {
    settings = new BenchmarkSettings(MODEL_TYPE::LITE);

    // TODO: learn how to process batches via tflite.
    // currently interpreter->tensor(input_index)->dims[0] = 1
    if (settings->batch_size != 1)
      throw string("Only BATCH_SIZE=1 is currently supported");
    
    session = new BenchmarkSession(settings);

    cout << "\nLoading graph..." << endl;

    model = tflite::FlatBufferModel::BuildFromFile(settings->graph_file().c_str());
    if (!model)
      throw "Failed to load graph from file " + settings->graph_file();

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter)
      throw string("Failed to construct interpreter");
    if (interpreter->AllocateTensors() != kTfLiteOk)
      throw string("Failed to allocate tensors");

    interpreter->SetNumThreads(settings->number_of_threads());

    int input_index = interpreter->inputs()[0];
    int output_index = interpreter->outputs()[0];
    auto input_type = interpreter->tensor(input_index)->type;
    auto output_type = interpreter->tensor(output_index)->type;
    if (input_type != output_type)
      throw format("Type of graph's input (%d) does not match type of its output (%d).",
                    int(input_type), int(output_type));

    switch (input_type) {
      case kTfLiteFloat32:
        if (settings->skip_internal_preprocessing)
          benchmark.reset(new TFLiteBenchmark<float, InCopy, OutCopy>(settings, interpreter.get(), input_index));
        else
          benchmark.reset(new TFLiteBenchmark<float, InNormalize, OutCopy>(settings, interpreter.get(), input_index));
        break;

      case kTfLiteUInt8:
        benchmark.reset(new TFLiteBenchmark<uint8_t, InCopy, OutDequantize>(settings, interpreter.get(), input_index));
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
    if (in_height != settings->image_size ||
        in_width != settings->image_size ||
        in_channels != settings->num_channels)
      throw format("Dimensions of graph's input do not correspond to dimensions of input image (%d*%d*%d*%d)",
                    settings->batch_size, settings->image_size, settings->image_size, settings->num_channels);

    TfLiteIntArray* out_dims = interpreter->tensor(output_index)->dims;
    int out_num = out_dims->data[0];
    int out_classes = out_dims->data[1];
    cout << format("Output tensor dimensions: %d*%d", out_num, out_classes) << endl;
    if (out_classes != settings->num_classes && out_classes != settings->num_classes+1)
      throw format("Unsupported number of classes in graph's output tensor. Supported numbers are %d and %d",
                    settings->num_classes, settings->num_classes+1);
    benchmark->has_background_class = out_classes == settings->num_classes+1;

  }

  ~Program() {
  }

  bool is_available_batch() {return session? session->get_next_batch(): false; }

  void LoadNextBatch() {
    session->get_next_batch();
    benchmark->load_images(session->batch_files());
  }

  void InferenceBatch() {
    if (interpreter->Invoke() != kTfLiteOk)
      throw "Failed to invoke tflite";
  }

  void UnloadBatch() {
    benchmark->save_results(session->batch_files());
  }

  const int batch_size() {return settings->batch_size; }
  const int batch_count() {return settings->batch_count; }

private:
  BenchmarkSettings *settings;
  BenchmarkSession *session;
  unique_ptr<IBenchmark> benchmark;
  unique_ptr<tflite::Interpreter> interpreter;
  unique_ptr<tflite::FlatBufferModel> model;
};


class SystemUnderTestSingleStream : public mlperf::SystemUnderTest {
public:
  SystemUnderTestSingleStream(Program *prg) : mlperf::SystemUnderTest() {
    this->prg = prg;
  };

  ~SystemUnderTestSingleStream() override = default;

  const std::string& Name() const override { return name_; }

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(samples.size());
    for (auto s : samples) {
      responses.push_back({s.id, 0, 0});
    }
    prg->InferenceBatch();
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
  }

  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {}

private:
  std::string name_{"SingleStreamSUT"};
  Program *prg;
};

class QuerySampleLibrarySingleStream : public mlperf::QuerySampleLibrary {
public:
  QuerySampleLibrarySingleStream(Program *prg) : mlperf::QuerySampleLibrary() {
    this->prg = prg;
  };

  ~QuerySampleLibrarySingleStream() = default;

  const std::string& Name() const override { return name_; }

  size_t TotalSampleCount() override { return prg->batch_count() * prg->batch_size(); }

  size_t PerformanceSampleCount() override { return prg->batch_size(); }

  void LoadSamplesToRam( const std::vector<mlperf::QuerySampleIndex>& samples) override {
    
    prg->LoadNextBatch();
    
    return;
  }

  void UnloadSamplesFromRam( const std::vector<mlperf::QuerySampleIndex>& samples) override {
    
    prg->UnloadBatch();

    return;
  }

private:
  std::string name_{"SingleStreamQSL"};
  Program *prg;
};

void TestSingleStream(Program *prg) {
  SystemUnderTestSingleStream sut(prg);
  QuerySampleLibrarySingleStream qsl(prg);

  mlperf::LogSettings log_settings;
  log_settings.log_output.prefix_with_datetime = true;

  mlperf::TestSettings ts;

  mlperf::StartTest(&sut, &qsl, ts, log_settings);
}

int main(int argc, char* argv[]) {
  try {
    Program *prg = new Program();
    TestSingleStream(prg);
    delete prg;
  }
  catch (const string& error_message) {
    cerr << "ERROR: " << error_message << endl;
    return -1;
  }
  return 0;
}
