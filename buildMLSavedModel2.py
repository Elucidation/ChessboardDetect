# coding=utf-8
import glob
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays

input_model_dir = './training_pipeline/training_models/run1_512_256_128_orig_5k_dataset_10000'
export_dir = 'ml/model/run97pct'

# builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  # Set up model
  feature_img = tf.feature_column.numeric_column("x", shape=[21,21], dtype=tf.uint8)

  # units = [1024,512,256]
  units = [512,256,128]
  estimator = tf.estimator.DNNClassifier(
    feature_columns=[feature_img],
    hidden_units=units,
    n_classes=2,
    model_dir=input_model_dir
    )

  def serving_input_receiver_fn():
    """Build the serving inputs."""
    # The outer dimension (None) allows us to batch up inputs for
    # efficiency. However, it also means that if we want a prediction
    # for a single instance, we'll need to wrap it in an outer list.
    inputs = {"x": tf.placeholder(shape=[None, 21, 21], dtype=tf.uint8)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

  estimator.export_savedmodel(export_dir, serving_input_receiver_fn)