# coding=utf-8
import tensorflow as tf
import os
from model import cnn_model_fn
from argparse import ArgumentParser

def main(args):
  export_dir = args.export_dir
  if not export_dir:
    export_dir = 'exported_models/%s' % (args.model_dir[args.model_dir.rstrip('/').rfind('/')+1:])
    
  print('Loading from %s ...' % args.model_dir)
  print('Exporting to %s ...' % export_dir)

  with tf.Session(graph=tf.Graph()) as sess:
    # Load CNN estimator from model_dir
    estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=args.model_dir,
    params={
      # CNN filter layers 1 and 2, and then dense layer # of units
      'filter_sizes':args.filter_sizes
    })

    # Build input function.
    image = tf.placeholder(tf.uint8, [None, 21, 21])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
      'x': image,
    })

    # Export model.
    estimator.export_savedmodel(export_dir, input_fn)

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--model_dir", dest="model_dir", required=True,
                      help="Directory to load model from (Ex. 'ml/model/cnn1').")
  parser.add_argument("--export_dir", dest="export_dir",
                      help="Directory to export model to, default uses model_dir name.")
  parser.add_argument("-fs", "--filter_sizes", dest="filter_sizes", nargs='+', type=int,
                      default=[32, 64, 1024], help="CNN model filter sizes")

  args = parser.parse_args()
  main(args)