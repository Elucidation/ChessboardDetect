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

export_dir = 'ml/model/001'

# builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  # Set up model
  featureA = tf.feature_column.numeric_column("x", shape=[11,11], dtype=tf.uint8)

  estimator = tf.estimator.DNNClassifier(
    feature_columns=[featureA],
    hidden_units=[256, 32],
    n_classes=2,
    dropout=0.1,
    model_dir='./xcorner_model_6k',
    )

  def serving_input_receiver_fn():
    """Build the serving inputs."""
    # The outer dimension (None) allows us to batch up inputs for
    # efficiency. However, it also means that if we want a prediction
    # for a single instance, we'll need to wrap it in an outer list.
    inputs = {"x": tf.placeholder(shape=[None, 11, 11], dtype=tf.uint8)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

  estimator.export_savedmodel(export_dir, serving_input_receiver_fn)

#   prediction_signature = signature_def_utils.build_signature_def(
#                         inputs={'x': tensor_info_x},
#                         outputs={'output': tensor_info_y},
#                         method_name=signature_constants.PREDICT_METHOD_NAME)
  
#   builder.add_meta_graph_and_variables(sess,
#                                        [tag_constants.TRAINING],
#                                        signature_def_map=
#                         {
#                           'predict_images':
#                               prediction_signature,
#                           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                               classification_signature,
#                         })

# # Add a MetaGraphDef for inference.
# with tf.Session(graph=tf.Graph()) as sess:
#   builder.add_meta_graph([tag_constants.SERVING])

# builder.save()