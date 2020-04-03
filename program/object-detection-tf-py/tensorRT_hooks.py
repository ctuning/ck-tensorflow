'''
hooks for using tensorRT with the object detection program.
names and parameters are defined as required by the detect.py infrastructure.

'''

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

def load_graph_tensorrt(params):
  graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v1.gfile.GFile(params["FROZEN_GRAPH"], 'rb') as f:
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


##no more needed
def convert_from_tensorrt(tmp_output_dict ):
  return tmp_output_dict


### names of tensors are different from normal TF names, but can be retrieved and a dict with the same shape of the original one can be formed, thus avoiding the conversion after the postprocessing.
# note that for the tf session, the names are enough and there is no real need to get the tensors.
def get_handles_to_tensors_RT():

  graph = tf.get_default_graph()
  tensor_dict = {}
  tensor_dict['num_detections'] = graph.get_tensor_by_name('import/num_detections:0')
  tensor_dict['detection_classes']=graph.get_tensor_by_name( 'import/detection_classes:0')
  tensor_dict['detection_boxes'] = graph.get_tensor_by_name('import/detection_boxes:0')
  tensor_dict['detection_scores'] = graph.get_tensor_by_name('import/detection_scores:0')

  image_tensor =graph.get_tensor_by_name('import/image_tensor:0')
  return tensor_dict, image_tensor

