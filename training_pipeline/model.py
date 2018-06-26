# CNN model, based off of the Tensorflow CNN Mnist Classifier tutorial.
import tensorflow as tf

def cnn_model(features, labels, mode, params):
  """Model function for CNN."""
  # Grayscale winsize=10 (21x21)

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 21, 21, 1])
  input_layer = tf.cast(input_layer, tf.float32)
  # Convert batch of images from uint8 to float64 normalized.
  # input_layer = tf.map_fn(lambda img: tf.image.per_image_standardization(img), input_layer)

  if labels is not None:
    bool_labels = tf.cast(labels, tf.bool)
    tf.summary.image('Input_Good', 
      tf.boolean_mask(input_layer, bool_labels), max_outputs=10)
    tf.summary.image('Input_Bad', 
      tf.boolean_mask(input_layer, tf.logical_not(bool_labels)), max_outputs=10)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=params['filter_sizes'][0],
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=params['filter_sizes'][1],
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * params['filter_sizes'][1]])
  dense = tf.layers.dense(inputs=pool2_flat, units=params['filter_sizes'][2], activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)
  return logits

def cnn_model_rgb(features, labels, mode, params):
  """Model function for CNN."""
  # RGB winsize=10 (Nx15x15x3)

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 15, 15, 3], name='ReshapeinModel1')
  input_layer = tf.cast(input_layer, tf.float32)
  # Convert batch of images from uint8 to float64 normalized.
  # input_layer = tf.map_fn(lambda img: tf.image.per_image_standardization(img), input_layer)

  tf.summary.image('Input', input_layer, max_outputs=5)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=params['filter_sizes'][0],
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  tf.summary.image('pool1', pool1[:,:,:,:4], max_outputs=5)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=params['filter_sizes'][1],
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  tf.summary.image('pool2', pool2[:,:,:,:4], max_outputs=5)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 3 * 3 * params['filter_sizes'][1]], name='ReshapePool2_flat')
  dense = tf.layers.dense(inputs=pool2_flat, units=params['filter_sizes'][2], activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)
  return logits


def cnn_model_rgb_small(features, labels, mode, params):
  """Model function for CNN."""
  # RGB winsize=10 (Nx15x15x3)

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 15, 15, 3], name='ReshapeinModel1')
  input_layer = tf.cast(input_layer, tf.float32)
  # Convert batch of images from uint8 to float64 normalized.
  # input_layer = tf.map_fn(lambda img: tf.image.per_image_standardization(img), input_layer)
  tf.summary.image('Input', input_layer, max_outputs=5)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=params['filter_sizes'][0],
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  tf.summary.image('pool1', pool1[:,:,:,:4], max_outputs=5)

  # Dense Layer
  pool1_flat = tf.reshape(pool1, [-1, 7 * 7 * params['filter_sizes'][0]], name='ReshapePool1_flat')
  dense = tf.layers.dense(inputs=pool1_flat, units=params['filter_sizes'][2], activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)
  return logits

def cnn_model_small(features, labels, mode, params):
  """Model function for CNN."""
  # Assumes 21x21 input size

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 21, 21, 1])
  input_layer = tf.cast(input_layer, tf.float32)

  if labels is not None:
    bool_labels = tf.cast(labels, tf.bool)
    tf.summary.image('Input_Good', 
      tf.boolean_mask(input_layer, bool_labels), max_outputs=10)
    tf.summary.image('Input_Bad', 
      tf.boolean_mask(input_layer, tf.logical_not(bool_labels)), max_outputs=10)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=params['filter_sizes'][0],
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool1_flat = tf.reshape(pool1, [-1, 10 * 10 * params['filter_sizes'][0]])
  dense = tf.layers.dense(inputs=pool1_flat, units=params['filter_sizes'][1], activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)
  return logits

def cnn_model_ultrasmall(features, labels, mode, params):
  """Model function for CNN."""
  # Assumes 15x15 input size

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 15, 15, 1])
  input_layer = tf.cast(input_layer, tf.float32)

  if labels is not None:
    bool_labels = tf.cast(labels, tf.bool)
    tf.summary.image('Input_Good', 
      tf.boolean_mask(input_layer, bool_labels), max_outputs=5)
    tf.summary.image('Input_Bad', 
      tf.boolean_mask(input_layer, tf.logical_not(bool_labels)), max_outputs=5)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=params['filter_sizes'][0],
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  tf.summary.image('pool1', pool1[:,:,:,:3], max_outputs=5)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=params['filter_sizes'][1],
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  tf.summary.image('pool2', pool2[:,:,:,:3], max_outputs=5)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 3 * 3 * params['filter_sizes'][1]])
  dense = tf.layers.dense(inputs=pool2_flat, units=params['filter_sizes'][2], activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)
  return logits

def cnn_model_big(features, labels, mode, params):
  """Model function for CNN."""
  channels = 2 # r g b gx gy

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 15, 15, channels])
  input_layer = tf.cast(input_layer, tf.float32)
  # Convert batch of images from uint8 to float64 normalized.
  # Note, potentially expensive
  # input_layer = tf.map_fn(lambda img: tf.image.per_image_standardization(img), input_layer)

  if labels is not None:
    bool_labels = tf.cast(labels, tf.bool)
    # tf.summary.image('Input_Good', 
    #   tf.boolean_mask(input_layer[:,:,:,:3], bool_labels), max_outputs=10)
    # tf.summary.image('Gx Good', 
    #   tf.boolean_mask(tf.expand_dims(input_layer[:,:,:,3], axis=-1), bool_labels), max_outputs=10)
    # tf.summary.image('Gy Good', 
    #   tf.boolean_mask(tf.expand_dims(input_layer[:,:,:,4], axis=-1), bool_labels), max_outputs=10)
    # tf.summary.image('Input_Bad', 
    #   tf.boolean_mask(input_layer[:,:,:,:3], tf.logical_not(bool_labels)), max_outputs=10)
    tf.summary.image('Gx Good', 
      tf.boolean_mask(tf.expand_dims(input_layer[:,:,:,0], axis=-1), bool_labels), max_outputs=10)
    tf.summary.image('Gy Good', 
      tf.boolean_mask(tf.expand_dims(input_layer[:,:,:,1], axis=-1), bool_labels), max_outputs=10)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=params['filter_sizes'][0],
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  # 31x31x3 output
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  # 15x15x3 output

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=params['filter_sizes'][1],
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  # 15x15x3 input, 15x15xparams['filter_sizes'][1] output
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  # 7x7xparams['filter_sizes'][1] output

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 3 * 3 * params['filter_sizes'][1]])
  dense = tf.layers.dense(inputs=pool2_flat, units=params['filter_sizes'][2], activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)
  return logits


def dnn_model_rgb(features, labels, mode, params):
  """Model function for DNN."""
  # RGB winsize=10 (Nx15x15x3)

  # Input Layer
  input_layer = tf.cast(features['x'], tf.float32)
  # Convert batch of images from uint8 to float64 normalized.
  # input_layer = tf.map_fn(lambda img: tf.image.per_image_standardization(img), input_layer)

  tf.summary.image('Input', input_layer, max_outputs=4)

  # Dense Layer #1
  dense1 = tf.layers.dense(
      inputs=input_layer,
      units=params['filter_sizes'][0],
      activation=tf.nn.relu)

  # Dense Layer #2
  dense2 = tf.layers.dense(
      inputs=dense1,
      units=params['filter_sizes'][1],
      activation=tf.nn.relu)

  # Dense Layer #3
  dense3 = tf.layers.dense(
    inputs=dense2,
    units=params['filter_sizes'][2],
    activation=tf.nn.relu)
  
  dropout = tf.layers.dropout(
      inputs=dense3, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Flatten input Nx15x15xparams['filter_sizes'][2]
  dropout_flat = tf.reshape(dropout, [-1, 15*15*params['filter_sizes'][2]], name='flat_dropout')

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout_flat, units=2)
  return logits

def cnn_model_rgb_v2(features, labels, mode, params):
  """Model function for CNN."""
  # RGB winsize=10 (Nx15x15x3)

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 15, 15, 3], name='ReshapeinModel1')
  input_layer = tf.cast(input_layer, tf.float32)
  # Convert batch of images from uint8 to float64 normalized.
  # input_layer = tf.map_fn(lambda img: tf.image.per_image_standardization(img), input_layer)

  tf.summary.image('Input', input_layer, max_outputs=4)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=128,
      kernel_size=[3, 3],
      activation=tf.nn.relu)
  tf.summary.image('conv1', conv1[:,:,:,:3], max_outputs=4)
  # Nx13x13x128

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=64,
      kernel_size=[3, 3],
      activation=tf.nn.relu)
  tf.summary.image('conv2', conv2[:,:,:,:3], max_outputs=4)
  # Nx11x11x64

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  # Nx5x5x64

  # Dense Layer 1
  pool1_flat = tf.reshape(pool1, [-1, 5 * 5 * 64])
  dense1 = tf.layers.dense(inputs=pool1_flat, units=128, activation=tf.nn.relu)
  
  # Dense Layer 2
  dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)
  dropout1 = tf.layers.dropout(
      inputs=dense2, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout1, units=2)
  return logits

def cnn_model_rgb_v3(features, labels, mode, params):
  """Model function for CNN."""
  # RGB winsize=10 (Nx15x15x3)

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 15, 15, 3], name='ReshapeinModel1')
  input_layer = tf.cast(input_layer, tf.float32)
  # Convert batch of images from uint8 to float64 normalized.
  # input_layer = tf.map_fn(lambda img: tf.image.per_image_standardization(img), input_layer)

  tf.summary.image('Input', input_layer, max_outputs=4)

  # Nx15x15xK
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  # tf.summary.image('conv1', conv1[:,:,:,:3], max_outputs=4)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  # tf.summary.image('conv2', conv2[:,:,:,:3], max_outputs=4)

  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  # tf.summary.image('conv3', conv3[:,:,:,:3], max_outputs=4)

  # Convolutional Layer #2
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  # tf.summary.image('conv4', conv4[:,:,:,:3], max_outputs=4)
  
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3)
  # Nx5x5xK
  # Convolutional Layer #1
  conv5 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2
  conv6 = tf.layers.conv2d(
      inputs=conv5,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  conv7 = tf.layers.conv2d(
      inputs=conv6,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2
  conv8 = tf.layers.conv2d(
      inputs=conv7,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  # tf.summary.image('conv4', conv4[:,:,:,:3], max_outputs=4)


  # Dense Layer 1
  conv8_flat = tf.reshape(conv8, [-1, 5 * 5 * 128])
  dense1 = tf.layers.dense(inputs=conv8_flat, units=512, activation=tf.nn.relu)
  dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout1, units=2)
  return logits


def cnn_model_fn(features, labels, mode, params):
  # logits = cnn_model(features, labels, mode, params)
  logits = cnn_model_rgb_v3(features, labels, mode, params)
  # logits = cnn_model_rgb_small(features, labels, mode, params)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  export_output = {'predict': tf.estimator.export.PredictOutput(predictions)}

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
      export_outputs=export_output)

  # If not predict, then  labels is not none
  labels = tf.cast(labels, tf.int32)
  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)