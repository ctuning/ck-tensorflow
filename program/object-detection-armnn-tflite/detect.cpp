/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#include "includes/detect.hpp"

using namespace std;
using namespace CK;

Settings settings;
BenchmarkSession session(&settings);

int main(int argc, char *argv[]) {
    try {
        init_benchmark();

        if (!settings.graph_file().c_str()) {
            throw string("Model file name is empty");
        }

        if (settings.batch_size() != 1)
            throw string("Only BATCH_SIZE=1 is currently supported");

        if (settings.verbose()) {
            armnn::ConfigureLogging(true, false, armnn::LogSeverity::Trace);
        } else if (settings.full_report()) {
            armnn::ConfigureLogging(true, false, armnn::LogSeverity::Info);
        } else {
            armnn::ConfigureLogging(true, false, armnn::LogSeverity::Error);
        }

        unique_ptr<IBenchmark> benchmark;
        armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
        armnn::NetworkId networkIdentifier;
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
        armnn::InputTensors inTensors;
        armnn::OutputTensors outTensors;

        // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc
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
            armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(settings.graph_file().c_str());

            if (!network)
                throw "Failed to load graph from file " + settings.graph_file();
            if (settings.verbose()) {
                cout << "Loaded model: " << settings.graph_file() << endl;
            }

            armnnTfLiteParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo(0, "normalized_input_image_tensor");
            armnnTfLiteParser::BindingPointInfo boxesBindingInfo = parser->GetNetworkOutputBindingInfo(0, "raw_outputs/box_encodings");
            armnnTfLiteParser::BindingPointInfo scoresBindingInfo = parser->GetNetworkOutputBindingInfo(0, "raw_outputs/class_predictions");


            armnn::TensorShape inShape = inputBindingInfo.second.GetShape();
            armnn::TensorShape boxesShape = boxesBindingInfo.second.GetShape();
            armnn::TensorShape scoresShape = scoresBindingInfo.second.GetShape();

            std::size_t inSize = inShape[0] * inShape[1] * inShape[2] * inShape[3];
            std::size_t boxesSize = boxesShape[0] * boxesShape[1] * boxesShape[2];
            std::size_t scoresSize = scoresShape[0] * scoresShape[1]* scoresShape[2];


            armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network, optOptions, runtime->GetDeviceSpec());
            if (optNet == nullptr) {
                cerr << endl << "\033[1;31mERROR:\033[0m" << endl
                     << "Fail to create a neural network from model with selected parameters..." << endl
                     << "(Probably NEON or OPENCL acceleration is not supported?)" << endl;
                exit(-1);
            }

            runtime->LoadNetwork(networkIdentifier, std::move(optNet));

            armnn::DataType input_type = inputBindingInfo.second.GetDataType();
            armnn::DataType boxes_type = boxesBindingInfo.second.GetDataType();
            armnn::DataType scores_type = scoresBindingInfo.second.GetDataType();


            void* input = input_type == armnn::DataType::Float32 ? (void*)new float[inSize] : (void*)new uint8_t[inSize];
            void* boxes = boxes_type == armnn::DataType::Float32 ? (void*)new float[boxesSize] : (void*)new uint8_t[boxesSize];
            void* scores = scores_type == armnn::DataType::Float32 ? (void*)new float[scoresSize] : (void*)new uint8_t[scoresSize];

            inTensors = MakeInputTensors(inputBindingInfo, input);
            outTensors = MakeOutputTensors(boxesBindingInfo, boxes);
            AddTensorToOutput(outTensors, scoresBindingInfo, scores);

            switch (input_type) {
                case armnn::DataType::Float32:
                    benchmark.reset(new ArmNNBenchmark<float, InNormalize, OutCopy>(&settings,
                                                                                    (float*)input,
                                                                                    (float*)boxes,
                                                                                    (float*)scores)
                                   );
                    break;

                case armnn::DataType::QuantisedAsymm8:
                    benchmark.reset(new ArmNNBenchmark<uint8_t, InCopy, OutDequantize>(&settings,
                                                                                       (uint8_t*)input,
                                                                                       (uint8_t*)boxes,
                                                                                       (uint8_t*)scores)
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
