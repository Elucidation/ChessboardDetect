# Training pipeline loading images from filepaths.
import tensorflow as tf
import numpy as np
import random
import preprocess
from argparse import ArgumentParser

def main(args):
  random.seed(100)
  # Load test dataset, used for evaluation
  test_folders = ['../datasets/test_datasetA',]
  test_imgs, test_labels, test_dataset_length, n_test_good = preprocess.loadMultipleDatapaths(test_folders, make_equal=True, pre_shuffle=True)
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
  
  train_imgs, train_labels, train_dataset_length, n_train_good = preprocess.loadMultipleDatapaths(train_folders, make_equal=True, pre_shuffle=True, max_count_each_entries=args.max_count_each_entries)
  print("Loaded train dataset '%s' : %d entries (%d good, %d bad)" % (train_folders, train_dataset_length, n_train_good, train_dataset_length - n_train_good ))

  # Build model.
  feature_img = tf.feature_column.numeric_column("x", shape=[21,21], dtype=tf.uint8)
  units = [512,256,128]
  model_str = "_".join(map(str,units))
  model_dir = './training_models/run1_%s_%s_dataset_%d' % (model_str, args.run_name, train_dataset_length)

  print("Using DNNClassifier %s : Output to %s" % (units, model_dir))
  estimator = tf.estimator.DNNClassifier(
    feature_columns=[feature_img],
    hidden_units=units,
    n_classes=2,
    # dropout=0.1,
    model_dir=model_dir,
    # model_dir='./training_models/lin_win%s' % (dataset_folder[dataset_folder.rfind('_')+1:]),
    )
  ############################
  # Train (For steps * train_steps = total steps)
  train_steps = 1000 
  for i in range(args.steps):
    # Test
    print("Training %d-%d/%d" % (i*train_steps,(i+1)*train_steps,args.steps*train_steps))
    estimator.train(input_fn=preprocess.input_fn(train_imgs, train_labels, train_dataset_length, is_training=True), steps=train_steps)

    # Evaluate
    metrics = estimator.evaluate(input_fn=preprocess.input_fn(test_imgs, test_labels))
    accuracy_score = metrics["accuracy"]
    print("-- Test Accuracy: {0:f}\n".format(accuracy_score))


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


  args = parser.parse_args()
  print(args)
  main(args)