#include "tensorflow/core/public/session.h" 

int main() {
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
  return 0;
}
