#include <dirent.h>
#include <chrono>

#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/framework/scope.h"
//#include "tensorflow/cc/ops/standard_ops.h"

using namespace std;

using tensorflow::Status;
using tensorflow::Tensor;

inline int getenv_i(const char *name, int def) {
  return getenv(name) ? atoi(getenv(name)) : def;
}

string graph_file(getenv("CK_ENV_TENSORFLOW_MODEL_FROZEN_FILE"));
string image_dir(getenv("CK_ENV_DATASET_IMAGENET_VAL"));
string image_file(getenv("CK_IMAGE_FILE"));
int batch_count = getenv_i("CK_BATCH_COUNT", 1);
int batch_size = getenv_i("CK_BATCH_SIZE", 1);
int image_count = batch_count * batch_size;
int image_size = getenv_i("CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT", 224);
float input_mean = 128.0f;
float input_std = 128.0f;

const int DT_FLOAT = 1;

/*
// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(string file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace   tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";
  auto file_reader = tensorflow::ops::ReadFile(root.WithOpName(input_name),
                                               file_name);
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                            DecodeJpeg::Channels(wanted_channels));
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
  return Status::OK();
}
*/
// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}


vector<string> get_image_list() {
  DIR *dir = opendir(image_dir.c_str());
  if (!dir) {
    cerr << "Unable to open images directory" << endl;
    exit(-1);
  }
  vector<string> img_files;
  int images_read = 0;
  struct dirent *ent;
  while ((ent = readdir(dir)) != NULL && images_read < image_count) {
    string file_name(ent->d_name);
    if (file_name == "." || file_name == "..")
      continue;
    img_files.push_back(file_name);
    images_read++;
  }
  closedir(dir);
  return img_files;
}

template <typename TData>
inline TData *get_random_raw_data(int data_count) {
  const int rnd_max = 1000000;
  const int rnd_range = 2 * rnd_max + 1;
  TData *data = new TData[data_count];
  for (int i = 0; i < data_count; ++i)
    data[i] = static_cast<TData>(-rnd_max + rand() % rnd_range) / static_cast<TData>(rnd_max);
  return data;
}

int main(int argc, char* argv[]) {
  cout << "Graph file: " << graph_file << endl;
  cout << "Image dir: " << image_dir << endl;
  cout << "Batch count: " << batch_count << endl;
  cout << "Batch size: " << batch_size << endl;
  
  string weights_dir(getenv("CK_ENV_TENSORFLOW_MODEL_WEIGHTS"));
  auto pos = weights_dir.find_last_of('/');
  if (pos == string::npos) {
    cerr << "Invalid model weights path" << endl;
    return -1;
  }
  weights_dir = weights_dir.substr(0, pos);
  cout << "Model weights dir: " << weights_dir << endl;
  
  // First we load and initialize the model.
  cout << "Loading frozen graph..." << endl;
  auto start_time = chrono::high_resolution_clock::now();  
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = weights_dir + '/' + graph_file;
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << "Loading graph failed: " << load_graph_status;
    return -1;
  }
  auto finish_time = chrono::high_resolution_clock::now(); 
  std::chrono::duration<double> elapsed = finish_time - start_time;
  cout << "Graph loaded in " << elapsed.count() << "s" << endl;
/*
  auto image_files = get_image_list();
  cout << "Images count: " << image_files.size() << endl;
  for (auto image_files : image_files)
    cout << image_files << endl;

  auto image_file = image_files[0];
*/
  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  /*std::vector<Tensor> resized_tensors;
  string image_path = image_dir + '/' + image_file;
  Status read_tensor_status = ReadTensorFromImageFile(
    image_path, image_size, image_size, input_mean, input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const Tensor& resized_tensor = resized_tensors[0];*/
  
/*
  tensorflow::SessionOptions options; 
  tensorflow::Session* session_pointer = nullptr;
  tensorflow::Status status = tensorflow::NewSession(options, &session_pointer);
  if (!status.ok()) {
    std::cout << "ERROR: " << status.ToString() << std::endl;
    return -1;
  }
  std::unique_ptr<tensorflow::Session> session(session_pointer);
  std::cout << "Session created." << std::endl;

  tensorflow::GraphDef tensorflow_graph;
  std::cout << "Graph created." << std::endl;

  std::cout << "Creating session..." << std::endl;
  status = session->Create(tensorflow_graph);
  if (!status.ok()) {
    std::cout << "Could not create TensorFlow Graph: " << status.ToString() << std::endl;
    return -1;
  }
*/
  float *in_data = get_random_raw_data<float>(2* 224 * 224 * 3);
  Tensor input(static_cast<tensorflow::DataType>(DT_FLOAT), tensorflow::TensorShape({2, 224, 224, 3}));
  memcpy(input.flat<float>().data(), in_data, 2 * 224 * 224 * 3 * sizeof(float));
  delete[] in_data;

  vector<Tensor> outputs;
  Status run_status = session->Run({{"input", input}},
                                   {"MobilenetV1/Predictions/Reshape_1"}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  auto output = outputs[0];

  int out_num = output.shape().dim_size(0);
  int out_height = output.shape().dim_size(1);
  int out_width = output.shape().dim_size(2);
  int out_channels = output.shape().dim_size(3);
  cout << "Output shape: " << out_num << " x "<< out_height << " x "<< out_width << " x "<< out_channels << endl;


  return 0;
}
