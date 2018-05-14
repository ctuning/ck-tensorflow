#include "tensorflow/core/public/session.h" 
#include <dirent.h>
#include <chrono>  

using namespace std;

using tensorflow::Status;

inline int getenv_i(const char *name, int def) {
  return getenv(name) ? atoi(getenv(name)) : def;
}

string graph_file(getenv("CK_ENV_TENSORFLOW_MODEL_FROZEN_FILE"));
string image_dir(getenv("CK_ENV_DATASET_IMAGENET_VAL"));
string image_file(getenv("CK_IMAGE_FILE"));
int batch_count = getenv_i("CK_BATCH_COUNT", 1);
int batch_size = getenv_i("CK_BATCH_SIZE", 1);
int image_count = batch_count * batch_size;


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
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  auto finish_time = chrono::high_resolution_clock::now(); 
  std::chrono::duration<double> elapsed = finish_time - start_time;
  cout << "Graph loaded in " << elapsed.count() << "s" << endl;

  auto img_files = get_image_list();
  cout << "Images count: " << img_files.size() << endl;
  for (auto img_file : img_files)
    cout << img_file << endl;
  return 0;
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

  return 0;
}
