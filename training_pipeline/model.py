# CNN model, based off of the Tensorflow CNN Mnist Classifier tutorial.
import tensorflow as tf

def cnn_model(features, labels, mode, params):
  """Model function for CNN."""

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 21, 21, 1])
  input_layer = tf.cast(input_layer, tf.float32)
  # Convert batch of images from uint8 to float64 normalized.
  # input_layer = tf.map_fn(lambda img: tf.image.per_image_standardization(img), input_layer)

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
  tf.summary.image('pool1', pool1[:,:,:,:3], max_outputs=5)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=params['filter_sizes'][1],
      kernel_size=[3, 3],
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


def cnn_model_fn(features, labels, mode, params):
  # logits = cnn_model(features, labels, mode, params)
  logits = cnn_model_ultrasmall(features, labels, mode, params)

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