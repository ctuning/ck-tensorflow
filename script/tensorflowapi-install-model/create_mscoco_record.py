import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

def get_label_map_dict(label_map_path):
    """Reads a label map and returns a dictionary of label names to id.
      Args:
        label_map_path: path to label_map.
      Returns:
        A dictionary mapping label names to id.
    """
    label_map = label_map_util.load_labelmap(label_map_path)
    label_map_dict = {}
    for item in label_map.item:
        label_map_dict[item.display_name.replace(" ", '')] = item.id
    return label_map_dict

def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
    ''' 
    Creates a TFRecord file from examples.

    Args:
        output_filename: Path to where output file is saved.
        label_map_dict: The label map dictionary.
        annotations_dir: Directory where annotation files are stored.
        image_dir: Directory where image files are stored.
        examples: Examples to parse and save to tf record.
    '''
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        path = os.path.join(annotations_dir, example + '.txt')

        if not os.path.exists(path):
            logging.warning('Could not find %s, ignoring example.', path)
            continue
    
        data = {}
        data['file_name'] = example
        data['objects'] = {}
        with tf.gfile.GFile(os.path.join(annotations_dir, example+'.txt'), 'r') as fid:
            annotation_str = fid.read()
            for i,object in enumerate(annotation_str.splitlines()):
                words = object.split(' ')
                data['objects'][str(i)] = {}
                data['objects'][str(i)]['name'] = words[0]
                data['objects'][str(i)]['bndbox'] = {'xmin' : words[4], 'ymin':words[5], 'width':words[6], 'height':words[7]}
                
        if os.path.isfile(os.path.join(image_dir, example + '.jpg')):
            with PIL.Image.open(os.path.join(image_dir, example + '.jpg')) as img:
                width, height = img.size 
                data['width'] = width 
                data['height'] = height 
                if img.format != 'JPEG':
                    continue
        else: 
            continue

        tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
        
        writer.write(tf_example.SerializeToString())

    writer.close()

def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory):
    """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['file_name'] is not a valid JPEG
    """
    img_path = os.path.join(image_subdirectory, data['file_name']+'.jpg')
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    width = int(data['width'])
    height = int(data['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    for idx, obj in data['objects'].iteritems():
        xmin_value = float(obj['bndbox']['xmin']) / width
        ymin_value = float(obj['bndbox']['ymin']) / height
        xmax_value = (float(obj['bndbox']['width']) + xmin_value) / width
        ymax_value = (float(obj['bndbox']['height']) + ymin_value) / height
        xmin.append(xmin_value)
        ymin.append(ymin_value)
        xmax.append(xmax_value)
        ymax.append(ymax_value)
        class_name = obj['name']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

    example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(
              data['file_name'].encode('utf8')),
          'image/source_id': dataset_util.bytes_feature(
              data['file_name'].encode('utf8')),
          'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
          'image/encoded': dataset_util.bytes_feature(encoded_jpg),
          'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))
    return example

def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = get_label_map_dict(FLAGS.label_map_path)
    
    logging.info('Reading from COCO dataset.')
    image_dir = os.path.join(data_dir, 'val2014')
    annotations_dir = os.path.join(data_dir, 'annotations','val2014')

    examples_list=[] 
    for file in os.listdir(annotations_dir):
        examples_list.append(file.split('.')[0])    

    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(num_examples * 0.7)
    num_val = num_examples - num_train
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

    train_output_path = os.path.join(FLAGS.output_dir, 'mscoco_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'mscoco_val.record')
    create_tf_record(train_output_path, label_map_dict, annotations_dir,
                   image_dir, train_examples)
    create_tf_record(val_output_path, label_map_dict, annotations_dir,
                   image_dir, val_examples)

if __name__ == '__main__':
    tf.app.run()
