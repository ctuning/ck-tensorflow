

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

def load_graph_tensorrt(params):
  graph_def = tf.compat.v1.GraphDef()
  with tf.gfile.GFile(params["FROZEN_GRAPH"], 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
  trt_graph = trt.create_inference_graph(
        input_graph_def=graph_def,
        outputs=['detection_boxes:0','detection_scores:0','detection_classes:0','num_detections:0'],
        max_batch_size=params["BATCH_SIZE"],
        max_workspace_size_bytes=4000000000,
        is_dynamic_op=True if params["TENSORRT_DYNAMIC"]==1 else False,
        precision_mode=params["TENSORRT_PRECISION"]
        )
  tf.import_graph_def(
        trt_graph,
        return_elements=['detection_boxes:0','detection_scores:0','detection_classes:0','num_detections:0'])


def convert_from_tensorrt(tmp_output_dict ):
  output_dict = {}
  output_dict['num_detections'] = tmp_output_dict[3]
  output_dict['detection_classes']= tmp_output_dict[2]
  output_dict['detection_boxes'] = tmp_output_dict[0]
  output_dict['detection_scores'] = tmp_output_dict[1]
  return output_dict



def get_handles_to_tensors_RT():

  graph = tf.get_default_graph()
  ops = graph.get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  return_elements=['import/detection_boxes:0','import/detection_scores:0','import/detection_classes:0','import/num_detections:0']
  tensor_dict = []
  for key in return_elements:
    if key in all_tensor_names:
      tensor_dict.append(graph.get_tensor_by_name(key))
  image_tensor =graph.get_tensor_by_name('import/image_tensor:0')
  return tensor_dict, image_tensor

