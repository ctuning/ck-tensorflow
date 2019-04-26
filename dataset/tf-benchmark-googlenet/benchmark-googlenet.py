#from builtins import range
from collections import namedtuple
from datetime import datetime
import csv
import math
import time

import tensorflow.python.platform
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 64,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('forward_only', False,
                            """Only run the forward pass.""")
tf.app.flags.DEFINE_boolean('forward_backward_only', False,
                            """Only run the forward-forward pass.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                           """The data format for Convnet operations.
                           Can be either NHWC or NCHW.
                           """)
tf.app.flags.DEFINE_string('csv_file', '',
                           """File to output timing information to in csv
                           format. If not file is passed in, csv file will
                           not be cteated.
                           """)

parameters = []

conv_counter = 1
pool_counter = 1
affine_counter = 1

TimingEntry = namedtuple(
    'TimingEntry', ['info_string', 'timestamp', 'num_batches', 'mean', 'sd'])

def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
    global conv_counter
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        if FLAGS.data_format == 'NCHW':
          strides = [1, 1, dH, dW]
        else:
          strides = [1, dH, dW, 1]
        conv = tf.nn.conv2d(inpOp, kernel, strides, padding=padType,
                            data_format=FLAGS.data_format)
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases,
                                         data_format=FLAGS.data_format),
                          conv.get_shape())
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        return conv1

def _affine(inpOp, nIn, nOut):
    global affine_counter
    global parameters
    name = 'affine' + str(affine_counter)
    affine_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        affine1 = tf.nn.relu_layer(inpOp, kernel, biases, name=name)
        parameters += [kernel, biases]
        return affine1

def _mpool(inpOp, kH, kW, dH, dW, padding):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    if FLAGS.data_format == 'NCHW':
      ksize = [1, 1, kH, kW]
      strides = [1, 1, dH, dW]
    else:
      ksize = [1, kH, kW, 1]
      strides = [1, dH, dW, 1]
    return tf.nn.max_pool(inpOp,
                          ksize=ksize,
                          strides=strides,
                          padding=padding,
                          data_format=FLAGS.data_format,
                          name=name)

def _apool(inpOp, kH, kW, dH, dW, padding):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    if FLAGS.data_format == 'NCHW':
      ksize = [1, 1, kH, kW]
      strides = [1, 1, dH, dW]
    else:
      ksize = [1, kH, kW, 1]
      strides = [1, dH, dW, 1]
    return tf.nn.avg_pool(inpOp,
                          ksize=ksize,
                          strides=strides,
                          padding=padding,
                          data_format=FLAGS.data_format,
                          name=name)

def _inception(inp, inSize, o1s, o2s1, o2s2, o3s1, o3s2, o4s1, o4s2):
    conv1 = _conv(inp, inSize, o1s, 1, 1, 1, 1, 'SAME')

    conv3_ = _conv(inp, inSize, o2s1, 1, 1, 1, 1, 'SAME')
    conv3 = _conv(conv3_, o2s1, o2s2, 3, 3, 1, 1, 'SAME')

    conv5_ = _conv(inp, inSize, o3s1, 1, 1, 1, 1, 'SAME')
    conv5 = _conv(conv5_, o3s1, o3s2, 5, 5, 1, 1, 'SAME')

    pool_ = _mpool(inp, o4s1, o4s1, 1, 1, 'SAME')
    pool = _conv(pool_, inSize, o4s2, 1, 1, 1, 1, 'SAME')

    if FLAGS.data_format == 'NCHW':
      channel_dim = 1
    else:
      channel_dim = 3
    incept = tf.concat([conv1, conv3, conv5, pool], channel_dim )
    return incept


def loss(logits, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels], 1 )
    onehot_labels = tf.sparse_to_dense(
        concated, tf.stack([batch_size, 1000]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=onehot_labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def inference(images):
    conv1 = _conv (images, 3, 64, 7, 7, 2, 2, 'SAME')
    pool1 = _mpool(conv1,  3, 3, 2, 2, 'SAME')
    conv2 = _conv (pool1,  64, 64, 1, 1, 1, 1, 'SAME')
    conv3 = _conv (conv2,  64, 192, 3, 3, 1, 1, 'SAME')
    pool3 = _mpool(conv3,  3, 3, 2, 2, 'SAME')

    incept3a = _inception(pool3,    192, 64, 96, 128, 16, 32, 3, 32)
    incept3b = _inception(incept3a, 256, 128, 128, 192, 32, 96, 3, 64)
    pool4 = _mpool(incept3b,  3, 3, 2, 2, 'SAME')
    incept4a = _inception(pool4,    480, 192,  96, 208, 16, 48, 3, 64)
    incept4b = _inception(incept4a, 512, 160, 112, 224, 24, 64, 3, 64)
    incept4c = _inception(incept4b, 512, 128, 128, 256, 24, 64, 3, 64)
    incept4d = _inception(incept4c, 512, 112, 144, 288, 32, 64, 3, 64)
    incept4e = _inception(incept4d, 528, 256, 160, 320, 32, 128, 3, 128)
    pool5 = _mpool(incept4e,  3, 3, 2, 2, 'SAME')
    incept5a = _inception(pool5,    832, 256, 160, 320, 32, 128, 3, 128)
    incept5b = _inception(incept5a, 832, 384, 192, 384, 48, 128, 3, 128)
    pool6 = _apool(incept5b,  7, 7, 1, 1, 'VALID')

    resh1 = tf.reshape(pool6, [-1, 1024])
    affn1 = _affine(resh1, 1024, 1000)

    return affn1


def time_tensorflow_run(session, target, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  if not isinstance(target, list):
    target = [target]
  target_op = tf.group(*target)
  for i in range(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target_op)
    duration = time.time() - start_time
    if i > num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))
  return TimingEntry(info_string, datetime.now(), FLAGS.num_batches, mn, sd)

def store_data_in_csv(timing_entries):
  with open(FLAGS.csv_file, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for timing_entry in timing_entries:
      writer.writerow(
          [timing_entry.info_string, timing_entry.timestamp,
           timing_entry.num_batches, timing_entry.mean, timing_entry.sd])

def run_benchmark(openme):
  global parameters
  timing_entries = []
  with tf.Graph().as_default():
    # Generate some dummy images.
    image_size = 224
    if FLAGS.data_format == 'NCHW':
      image_shape = [FLAGS.batch_size, 3, image_size, image_size]
    else:
      image_shape = [FLAGS.batch_size, image_size, image_size, 3]
    images = tf.Variable(tf.random_normal(image_shape,
                                          dtype=tf.float32,
                                          stddev=1e-1))

    labels = tf.Variable(tf.ones([FLAGS.batch_size],
                                 dtype=tf.int32))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    last_layer = inference(images)

    # Build an initialization operation.
    tf_major_ver = int(tf.__version__.split(".")[0])
    tf_minor_ver = int(tf.__version__.split(".")[1])
    if(tf_major_ver == 0 and tf_minor_ver < 12): # For tf version <0.12.0
      init = tf.initialize_all_variables()
    else: # For tf version >= 0.12.0
      init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session('')
    sess.run(init)

    run_forward = True
    run_forward_backward = True
    if FLAGS.forward_only and FLAGS.forward_backward_only:
      raise ValueError("Cannot specify --forward_only and "
                       "--forward_backward_only at the same time.")
    if FLAGS.forward_only:
      run_forward_backward = False
    elif FLAGS.forward_backward_only:
      run_forward = False

    if run_forward:
      # Run the forward benchmark.
      x=time_tensorflow_run(sess, last_layer, "Forward")
      openme['time_fw_norm']=x.mean
      timing_entries.append(x)

    if run_forward_backward:
      # Add a simple objective so we can calculate the backward pass.
      objective = loss(last_layer, labels)
      # Compute the gradient with respect to all the parameters.
      grad = tf.gradients(objective, parameters)
      # Run the backward benchmark.
      x=time_tensorflow_run(sess, grad, "Forward-backward")
      openme['time_fwbw_norm']=x.mean
      openme['execution_time']=x.mean
      timing_entries.append(x)

  if FLAGS.csv_file:
    store_data_in_csv(timing_entries)


def main(_):
  openme={}

  run_benchmark(openme)

  import json
  with open('tmp-ck-timer.json', 'w') as o:
     json.dump(openme, o)


if __name__ == '__main__':
  tf.app.run()
