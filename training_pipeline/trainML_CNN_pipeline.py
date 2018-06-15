# Training pipeline loading images from filepaths.
import tensorflow as tf
import numpy as np
import random
import preprocess
from argparse import ArgumentParser
from model import cnn_model_fn
from time import time

def main(args):
  random.seed(100)
  make_equal = not args.use_all_entries
  # Load test dataset, used for evaluation
  test_folders = ['../datasets/test_datasetA',]
  test_imgs, test_labels, test_dataset_length, n_test_good = preprocess.loadMultipleDatapaths(test_folders, make_equal=make_equal, pre_shuffle=True)
  if args.max_count_each_entries:
    print('Using %d max_count_each_entries' % args.max_count_each_entries)
  print("Loaded test dataset '%s' : %d entries (%d good, %d bad)" % (test_folders, test_dataset_length, n_test_good, test_dataset_length - n_test_good ))

  # Load training dataset(s).
  if args.training_folders:
    train_folders = args.training_folders
  else:
    # train_folders = ['../datasets/dataset_gray_10']
    allfiles = ['chess_beer.mp4', 'random1.mp4', 'match2.mp4','output.avi','output.mp4',
        'speedchess1.mp4','wgm_1.mp4','gm_magnus_1.mp4',
        'bro_1.mp4','output2.avi','john1.mp4','john2.mp4','swivel.mp4',
        'sam1.mp4', 'sam2.mp4',]
    train_folders = ['../datasets/dataset_gray_10']
    train_folders.extend(map(lambda x: '../results/%sam1_vidstream_frames/tiles' % x[:-4], allfiles ))
  
  train_imgs, train_labels, train_dataset_length, n_train_good = preprocess.loadMultipleDatapaths(train_folders, make_equal=make_equal, pre_shuffle=True, max_count_each_entries=args.max_count_each_entries)
  print("Loaded train dataset '%s' : %d entries (%d good, %d bad)" % (train_folders, train_dataset_length, n_train_good, train_dataset_length - n_train_good ))

  # Build model.
  feature_img = tf.feature_column.numeric_column("x", shape=[21,21], dtype=tf.uint8)
  model_dir = './training_models/cnn_adam_%s_dataset_%d' % (args.run_name, train_dataset_length)

  estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir,
    params={
      # CNN filter layers 1 and 2, and then dense layer # of units
      'filter_sizes':args.filter_sizes
    })

  print("Using CNN ChessXCorner Classifier, output to %s" % (model_dir))
  ############################
  # Train (For steps * train_steps = total steps)
  train_steps = 100
  for i in range(args.steps):
    # Test
    print("Training %d-%d/%d" % (i*train_steps,(i+1)*train_steps,args.steps*train_steps))
    ta = time()
    estimator.train(
      input_fn=preprocess.input_fn(train_imgs, train_labels, train_dataset_length, is_training=True),
      steps=train_steps)
    tb = time()

    # Evaluate
    metrics = estimator.evaluate(input_fn=preprocess.input_fn(test_imgs, test_labels))
    accuracy_score = metrics["accuracy"]
    print("-- Test Accuracy: {0:f}\n".format(accuracy_score))
    tc = time()
    print("Train took %g, Evaluate took %g" % (tb-ta, tc-tb))


  ############################
  # Validate
  # predictions = estimator.predict(input_fn=preprocess.input_fn(test_imgs, test_labels))

  # test_labels = np.array(f_labels[split:])

  # count_good = 0
  # for i,(prediction,true_answer) in enumerate(zip(predictions, test_labels.astype(np.int))):
  #   if prediction['probabilities'].argmax() != true_answer:
  #     # print("Failure on %d: %s" % (i, prediction['probabilities']))
  #     continue
  #   else:
  #     count_good += 1

  # success_rate = float(count_good) / len(test_labels)
  # print("Total %d/%d right ~ %.2f%% success rate" % (count_good, len(test_labels), 100*success_rate))
 
if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("-m", "--max_each", dest="max_count_each_entries", type=int,
                      help="Maximum count of good or bad tiles per each folder")
  parser.add_argument("--name", dest="run_name", required=True,
                      help="Name of model run name")
  parser.add_argument("-s", "--steps", dest="steps", required=True, type=int,
                      help="Number of eval steps to run (x1000 train steps).")
  parser.add_argument("training_folders", nargs='+',
                      help="Training folder tile paths used for training")
  parser.add_argument("-fs", "--filter_sizes", dest="filter_sizes", nargs='+', type=int,
                      default=[32, 64, 1024], help="CNN model filter sizes")
  parser.add_argument("-use_all_entries",
                      action='store_true', help="Whether to make input datasets equal good/bad.")


  args = parser.parse_args()
  print("Arguments passed: \n\t%s\n" % args)
  main(args)