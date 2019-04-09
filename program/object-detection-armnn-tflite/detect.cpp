/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#include <iomanip>
#include <vector>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"


#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"

#include "benchmark.h"

using namespace std;
using namespace CK;


template<typename TData, typename TInConverter, typename TOutConverter>
class TFLiteBenchmark : public Benchmark<TData, TInConverter, TOutConverter> {
public:
    TFLiteBenchmark(BenchmarkSettings *settings, tflite::Interpreter *interpreter, int input_index)
            : Benchmark<TData, TInConverter, TOutConverter>(
            settings, interpreter->typed_tensor<TData>(input_index),
            interpreter->typed_output_tensor<TData>(0),
            interpreter->typed_output_tensor<TData>(1),
            interpreter->typed_output_tensor<TData>(2),
            interpreter->typed_output_tensor<TData>(3)) {
    }
};

template <typename TData, typename TInConverter, typename TOutConverter>
class ArmNNBenchmark : public Benchmark<TData, TInConverter, TOutConverter> {
public:
    ArmNNBenchmark(BenchmarkSettings* settings,
                   TData *in_ptr,
                   TData *boxes_ptr,
                   TData *classes_ptr,
                   TData *scores_ptr,
                   TData *detections_count_ptr
                   )
            : Benchmark<TData, TInConverter, TOutConverter>(settings, in_ptr, boxes_ptr, classes_ptr, scores_ptr, detections_count_ptr) {
    }
};

armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
        armnn::TensorInfo>& input, const void* inputTensorData)
{
    return { {input.first, armnn::ConstTensor(input.second, inputTensorData) } };
}

armnn::OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId,
        armnn::TensorInfo>& output, void* outputTensorData)
{
    return { {output.first, armnn::Tensor(output.second, outputTensorData) } };
}

void AddTensorToOutput(armnn::OutputTensors &v, const std::pair<armnn::LayerBindingId,
        armnn::TensorInfo>& output, void* outputTensorData ) {
    v.push_back({output.first, armnn::Tensor(output.second, outputTensorData) });
}

int main(int argc, char *argv[]) {
    try {
        init_benchmark();

        BenchmarkSettings settings;

        if (!settings.graph_file().c_str()) {
            throw string("Model file name is empty");
        }

        if (settings.batch_size() != 1)
            throw string("Only BATCH_SIZE=1 is currently supported");

        BenchmarkSession session(&settings);

        //unique_ptr<IBenchmark> benchmark;
        unique_ptr<tflite::FlatBufferModel> model;
        unique_ptr<tflite::Interpreter> interpreter;

        unique_ptr<IBenchmark> benchmark;
        armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
        armnn::NetworkId networkIdentifier;
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
        armnn::InputTensors inTensors;
        armnn::OutputTensors outTensors;

        // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc
        //std::vector<armnn::BackendId> optOptions = {armnn::Compute::CpuAcc, armnn::Compute::GpuAcc};
        std::vector<armnn::BackendId> optOptions = {armnn::Compute::CpuRef};
        if( settings.use_neon() && settings.use_opencl()) {
	    if (settings.verbose()) {
		cout << "Enable CPU and GPU acceleration" << endl;
	    }
            optOptions = {armnn::Compute::CpuAcc, armnn::Compute::GpuAcc};
        } else if( settings.use_neon() ) {
	    if (settings.verbose()) {
		cout << "Enable CPU acceleration" << endl;
	    }
            optOptions = {armnn::Compute::CpuAcc};
        } else if( settings.use_opencl() ) {
	    if (settings.verbose()) {
		cout << "Enable GPU acceleration" << endl;
	    }
            optOptions = {armnn::Compute::GpuAcc};
        }

        cout << endl << "Loading graph..." << endl;
        measure_setup([&] {
            model = tflite::FlatBufferModel::BuildFromFile(settings.graph_file().c_str());
            if (!model)
                throw std::string("Failed to load graph from file ") + settings.graph_file().c_str();
            if (settings.verbose()) {
                cout << "Loaded model: " << settings.graph_file() << endl;
            }

            armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(settings.graph_file().c_str());
            if (!network)
                throw "Failed to load graph from file " + settings.graph_file();

            armnnTfLiteParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo(0, "normalized_input_image_tensor");
            armnnTfLiteParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess");
            armnnTfLiteParser::BindingPointInfo outputBindingInfo1 = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:1");
            armnnTfLiteParser::BindingPointInfo outputBindingInfo2 = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:2");
            armnnTfLiteParser::BindingPointInfo outputBindingInfo3 = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:3");

            armnn::TensorShape inShape = inputBindingInfo.second.GetShape();
            armnn::TensorShape outShape = outputBindingInfo.second.GetShape();
            armnn::TensorShape outShape1 = outputBindingInfo1.second.GetShape();
            armnn::TensorShape outShape2 = outputBindingInfo2.second.GetShape();
            armnn::TensorShape outShape3 = outputBindingInfo3.second.GetShape();
            std::size_t inSize = inShape[0] * inShape[1] * inShape[2] * inShape[3];
            std::size_t outSize = outShape[0] * outShape[1] * outShape[2];
            std::size_t outSize1 = outShape1[0] * outShape1[1];
            std::size_t outSize2 = outShape2[0] * outShape2[1];
            std::size_t outSize3 = outShape3[0];

            armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network, optOptions, runtime->GetDeviceSpec());

            runtime->LoadNetwork(networkIdentifier, std::move(optNet));

            armnn::DataType input_type = inputBindingInfo.second.GetDataType();
            armnn::DataType output_type = outputBindingInfo.second.GetDataType();
            armnn::DataType output1_type = outputBindingInfo1.second.GetDataType();
            armnn::DataType output2_type = outputBindingInfo2.second.GetDataType();
            armnn::DataType output3_type = outputBindingInfo3.second.GetDataType();

            void* input = input_type == armnn::DataType::Float32 ? (void*)new float[inSize] : (void*)new uint8_t[inSize];
            void* boxes = output_type == armnn::DataType::Float32 ? (void*)new float[outSize] : (void*)new uint8_t[outSize];
            void* classes = output1_type == armnn::DataType::Float32 ? (void*)new float[outSize1] : (void*)new uint8_t[outSize1];
            void* scores = output2_type == armnn::DataType::Float32 ? (void*)new float[outSize2] : (void*)new uint8_t[outSize2];
            void* detections_count = output3_type == armnn::DataType::Float32 ? (void*)new float[outSize3] : (void*)new uint8_t[outSize3];

            inTensors = MakeInputTensors(inputBindingInfo, input);
            outTensors = MakeOutputTensors(outputBindingInfo, boxes);
            AddTensorToOutput(outTensors, outputBindingInfo1, classes);
            AddTensorToOutput(outTensors, outputBindingInfo2, scores);
            AddTensorToOutput(outTensors, outputBindingInfo3, detections_count);

            switch (input_type) {
                case armnn::DataType::Float32:
                    benchmark.reset(new ArmNNBenchmark<float, InNormalize, OutCopy>(&settings,
                                                                                    (float*)input,
                                                                                    (float*)boxes,
                                                                                    (float*)classes,
                                                                                    (float*)scores,
                                                                                    (float*)detections_count)
                                   );
                    break;

                case armnn::DataType::QuantisedAsymm8:
                    benchmark.reset(new ArmNNBenchmark<uint8_t, InCopy, OutDequantize>(&settings,
                                                                                       (uint8_t*)input,
                                                                                       (uint8_t*)boxes,
                                                                                       (uint8_t*)classes,
                                                                                       (uint8_t*)scores,
                                                                                       (uint8_t*)detections_count)
                                   );
                    break;

                default:
                    throw format("Unsupported type of graph's input: %d. "
                                 "Supported types are: Float32 (%d), UInt8 (%d)",
                                 int(input_type), int(armnn::DataType::Float32), int(armnn::DataType::QuantisedAsymm8));
            }
        });

        cout << "\nProcessing batches..." << endl;
       measure_prediction([&] {
           while (session.get_next_batch()) {
               session.measure_begin();
               benchmark->load_images(session.batch_files());
               session.measure_end_load_images();

               session.measure_begin();
               if (runtime->EnqueueWorkload(networkIdentifier, inTensors, outTensors) != armnn::Status::Success)
                   throw "Failed to invoke the classifier";
               session.measure_end_prediction();

               session.measure_begin();
               benchmark->non_max_suppression(session.batch_files());
               session.measure_end_non_max_suppression();

               benchmark->save_results(session.batch_files());
           }
        });
    }
    catch (const string &error_message) {
        cerr << "ERROR: " << error_message << endl;
        return -1;
    }

    return 0;
}
