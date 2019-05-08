/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#include "includes/detect.hpp"

int main(int argc, char *argv[]) {
    try {
        init_benchmark();

        if (!settings.graph_file().c_str()) {
            throw string("Model file name is empty");
        }

        if (settings.batch_size() != 1)
            throw string("Only BATCH_SIZE=1 is currently supported");

        unique_ptr<IBenchmark> benchmark;
        unique_ptr<FlatBufferModel> model;
        unique_ptr<Interpreter> interpreter;

        cout << endl << "Loading graph..." << endl;
        measure_setup([&] {
            model = FlatBufferModel::BuildFromFile(settings.graph_file().c_str());
            if (!model)
                throw "Failed to load graph from file " + settings.graph_file();
            if (settings.verbose()) {
                cout << "Loaded model " << settings.graph_file() << endl;
                model->error_reporter();
                cout << "resolved reporter" << endl;
                cout << endl << "Number of threads: " << settings.number_of_threads() << endl;
            }

            ops::builtin::BuiltinOpResolver resolver;
            if (!settings.default_model_settings()) resolver.AddCustom("TFLite_Detection_PostProcess", Register_Postprocess_with_NMS());

            InterpreterBuilder(*model, resolver)(&interpreter);
            if (!interpreter)
                throw string("Failed to construct interpreter");
            if (interpreter->AllocateTensors() != kTfLiteOk)
                throw string("Failed to allocate tensors");

            int input_size = interpreter->inputs().size();
            int output_size = interpreter->outputs().size();

            if (settings.verbose()) {
                cout << "tensors size: " << interpreter->tensors_size() << endl;
                cout << "nodes size: " << interpreter->nodes_size() << endl;
                cout << "number of inputs: " << input_size << endl;
                cout << "number of outputs: " << output_size << endl;
                cout << "input(0) name: " << interpreter->GetInputName(0) << endl;

                int t_size = interpreter->tensors_size();
                for (int i = 0; i < t_size; i++) {
                    if (interpreter->tensor(i)->name)
                        cout << i << ": " << interpreter->tensor(i)->name << ", "
                             << interpreter->tensor(i)->bytes << ", "
                             << interpreter->tensor(i)->type << ", "
                             << interpreter->tensor(i)->params.scale << ", "
                             << interpreter->tensor(i)->params.zero_point << endl;
                }
            }

            interpreter->SetNumThreads(settings.number_of_threads());

            if (settings.verbose()) PrintInterpreterState(interpreter.get());

            int input_index = interpreter->inputs()[0];
            int detection_boxes_id = interpreter->outputs()[0];
            int detection_classes_id = interpreter->outputs()[1];
            int detection_scores_id = interpreter->outputs()[2];
            auto input_type = interpreter->tensor(input_index)->type;

            switch (input_type) {
                case kTfLiteFloat32:
                    benchmark.reset(new TFLiteBenchmark<float, InNormalize, OutCopy>(&settings, interpreter.get(),
                                                                                     input_index));
                    break;

                case kTfLiteUInt8:
                    benchmark.reset(new TFLiteBenchmark<uint8_t, InCopy, OutDequantize>(&settings, interpreter.get(),
                                                                                        input_index));
                    break;

                default:
                    throw format("Unsupported type of graph's input: %d. "
                                 "Supported types are: Float32 (%d), UInt8 (%d)",
                                 int(input_type), int(kTfLiteFloat32), int(kTfLiteUInt8));
            }

            TfLiteIntArray *in_dims = interpreter->tensor(input_index)->dims;
            int in_num = in_dims->data[0];
            int in_height = in_dims->data[1];
            int in_width = in_dims->data[2];
            int in_channels = in_dims->data[3];

            if (in_height != settings.image_size_height() ||
                in_width != settings.image_size_width() ||
                in_channels != settings.num_channels())
                throw format("Dimensions of graph's input do not correspond to dimensions of input image (%d*%d*%d*%d)",
                             settings.batch_size(),
                             settings.image_size_height(),
                             settings.image_size_width(),
                             settings.num_channels());

            int frames = 1;
            TfLiteIntArray *detection_boxes_ptr = interpreter->tensor(detection_boxes_id)->dims;
            int boxes_count = detection_boxes_ptr->data[1];
            int boxes_length = detection_boxes_ptr->data[2];

            TfLiteIntArray *detection_classes_ptr = interpreter->tensor(detection_classes_id)->dims;
            int classes_count = detection_classes_ptr->data[1];

            TfLiteIntArray *detection_scores_ptr = interpreter->tensor(detection_scores_id)->dims;
            int scores_count = detection_scores_ptr->data[1];

            if (settings.verbose()) {
                cout << format("Input tensor dimensions (NHWC): %d*%d*%d*%d", in_num, in_height, in_width, in_channels)
                     << endl;
                cout << format("Detection boxes tensor dimensions: %d*%d*%d", frames, boxes_count, boxes_length)
                     << endl;
                cout << format("Detection classes tensor dimensions: %d*%d", frames, classes_count) << endl;
                cout << format("Detection scores tensor dimensions: %d*%d", frames, scores_count) << endl;
                cout << format("Number of detections tensor dimensions: %d*1", frames) << endl;
            }

        });

        cout << "\nProcessing batches..." << endl;
        measure_prediction([&] {
            while (session.get_next_batch()) {
                session.measure_begin();
                benchmark->load_images(session.batch_files());
                session.measure_end_load_images();

                session.measure_begin();
                if (interpreter->Invoke() != kTfLiteOk)
                    throw "Failed to invoke tflite";
                session.measure_end_prediction();

                benchmark->export_results(session.batch_files());

                benchmark->save_results(session.batch_files());
            }
        });

        finish_benchmark(session);
    }
    catch (const string &error_message) {
        cerr << "ERROR: " << error_message << endl;
        return -1;
    }

    return 0;
}
