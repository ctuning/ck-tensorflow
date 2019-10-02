/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#include <future>
#include <algorithm> 
#include <numeric> 

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

  //bool is_available_batch() {return session? session->get_next_batch(): false; }

  void LoadNextBatch(const std::vector<mlperf::QuerySampleIndex>& img_indices) {
    std::cout << "LoadNextBatch([";
    for( auto idx : img_indices) {
      std::cout << idx << ' ';
    }
    std::cout << "])" << std::endl;
    session->load_filenames(img_indices);
    benchmark->load_images( session );
  }

  int InferenceOnce(int img_idx) {
    benchmark->get_random_image( img_idx );
    if (interpreter->Invoke() != kTfLiteOk)
      throw "Failed to invoke tflite";
    return benchmark->get_next_result();
  }

  void UnloadBatch(const std::vector<mlperf::QuerySampleIndex>& img_indices) {
    //benchmark->save_results( );
  }

  const int available_images_max() { return settings->list_of_available_imagefiles().size(); }
  const int images_in_memory_max() { return settings->images_in_memory_max; }

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

    std::cout << "IssueQuery([" << samples.size() << "]," << samples[0].id << "," << samples[0].index << ")" << std::endl;

    // This is currently a completely fake response, only to satisfy the interface
    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(samples.size());
    float encoding_buffer[samples.size()];
    int i=0;
    for (auto s : samples) {
      int predicted_class = prg->InferenceOnce(s.index);
      std::cout << "Query image index: " << s.index << " -> Predicted class: " << predicted_class << std::endl;


      /* This would be the correct way to pass in one integer index:
      */
//      int single_value_buffer[] = { (int)predicted_class };

      /* This conversion is subtly but terribly wrong
         yet we use it here in order to use Guenther's parsing script:
      */
      encoding_buffer[i] = (float)predicted_class;
      responses.push_back({s.id, uintptr_t(&encoding_buffer[i]), sizeof(encoding_buffer[i])});
      ++i;
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
  }

  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {

    int size = latencies_ns.size();
    std::vector<mlperf::QuerySampleLatency> v(size);
    std::copy (latencies_ns.begin(), latencies_ns.end(), v.begin() );

    cout << endl << "------------------------------------------------------------";
    cout << endl << "|            LATENCIES (in nanoseconds and fps)            |";
    cout << endl << "------------------------------------------------------------";
    sort(v.begin(), v.end());
    long avg = long(accumulate(v.begin(), v.end(), 0L))/size;
    int p50 = size * 0.5;
    int p90 = size * 0.9;
    cout << endl << "Number of queries run: " << size;
    cout << endl << "Min latency:                      " << v[0]            << "ns  (" << 1e9/v[0]            << " fps)";
    cout << endl << "Median latency:                   " << v[p50]          << "ns  (" << 1e9/v[p50]          << " fps)";
    cout << endl << "Average latency:                  " << avg             << "ns  (" << 1e9/avg             << " fps)";
    cout << endl << "90 percentile latency:            " << v[p90]          << "ns  (" << 1e9/v[p90]          << " fps)";
    cout << endl << "First query (cold model) latency: " << latencies_ns[0] << "ns  (" << 1e9/latencies_ns[0] << " fps)";
    cout << endl << "Max latency:                      " << v[size-1]       << "ns  (" << 1e9/v[size-1]       << " fps)";

    cout << endl << "------------------------------------------------------------ " << endl;
  }

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

  size_t TotalSampleCount() override { return prg->available_images_max(); }

  size_t PerformanceSampleCount() override { return prg->images_in_memory_max(); }

  void LoadSamplesToRam( const std::vector<mlperf::QuerySampleIndex>& samples) override {
    prg->LoadNextBatch(samples);
    return;
  }

  void UnloadSamplesFromRam( const std::vector<mlperf::QuerySampleIndex>& samples) override {
    prg->UnloadBatch(samples);
    return;
  }

private:
  std::string name_{"SingleStreamQSL"};
  Program *prg;
};

void TestSingleStream(Program *prg) {
  SystemUnderTestSingleStream sut(prg);
  QuerySampleLibrarySingleStream qsl(prg);

  const std::string config_file_path = getenv_s("CK_ENV_MLPERF_INFERENCE_V05") + "/mlperf.conf";

  std::string guenther_model_name = "anything_else";
  if( getenv_s("CK_ENV_TENSORFLOW_MODEL_TFLITE_FILENAME") == "resnet50_v1.tflite" )
    guenther_model_name = "resnet50";

  const std::string scenario_string = getenv_s("CK_LOADGEN_SCENARIO");
  const std::string mode_string = getenv_s("CK_LOADGEN_MODE");

  std::cout << "Config path: " << config_file_path << std::endl;
  std::cout << "Guenther Model Name: " << guenther_model_name << std::endl;
  std::cout << "LoadGen Scenario: " << scenario_string << std::endl;
  std::cout << "LoadGen Mode: " << mode_string << std::endl;

  mlperf::TestSettings ts;

  // This should have been done automatically inside ts.FromConfig() !
  ts.scenario = ( scenario_string == "SingleStream")    ? mlperf::TestScenario::SingleStream
              : ( scenario_string == "MultiStream")     ? mlperf::TestScenario::MultiStream
              : ( scenario_string == "MultiStreamFree") ? mlperf::TestScenario::MultiStreamFree
              : ( scenario_string == "Server")          ? mlperf::TestScenario::Server
              : ( scenario_string == "Offline")         ? mlperf::TestScenario::Offline : mlperf::TestScenario::SingleStream;

  ts.mode     = ( mode_string == "SubmissionRun")       ? mlperf::TestMode::SubmissionRun
              : ( mode_string == "AccuracyOnly")        ? mlperf::TestMode::AccuracyOnly
              : ( mode_string == "PerformanceOnly")     ? mlperf::TestMode::PerformanceOnly
              : ( mode_string == "FindPeakPerformance") ? mlperf::TestMode::FindPeakPerformance : mlperf::TestMode::SubmissionRun;

  if (ts.FromConfig(config_file_path, guenther_model_name, scenario_string)) {
    std::cout << "Issue with config file " << config_file_path << std::endl;
    exit(1);
  }

  //ts.min_query_count = std::min( prg->available_images_max(), prg->images_in_memory_max() )*2;
  //ts.max_query_count = 20;
  //ts.min_duration_ms = 0;
  //ts.max_duration_ms = 2000;

  mlperf::LogSettings log_settings;
  log_settings.log_output.prefix_with_datetime = false;
  log_settings.enable_trace = false;

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
