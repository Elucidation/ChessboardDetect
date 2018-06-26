# Training pipeline loading images from tfrecords
import tensorflow as tf
import numpy as np
import random
import glob
import preprocess
from argparse import ArgumentParser
from model import cnn_model_fn
from time import time

def randomize_image(img, contrast_range=[0.9,1.1], brightness_max=0.5):
  # Apply random flips/rotations and contrast/brightness changes to image
  img = tf.image.random_flip_left_right(img)
  img = tf.image.random_flip_up_down(img)
  img = tf.image.random_contrast(img, lower=contrast_range[0], upper=contrast_range[1])
  img = tf.image.random_brightness(img, max_delta=brightness_max)
  img = tf.image.random_hue(img, max_delta=0.1)
  img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
  img = tf.contrib.image.rotate(img, tf.random_uniform([1], minval=-np.pi, maxval=np.pi))
  return img

def add_image_gradient(img):
  # Add image gradient x and y to the 3rd dimension
  # Input is 21x21x3, output should be 21x21x5
  # gx 21x21x3 for each color channel
  # gx = tf.concat([tf.zeros([1,img.shape[1],img.shape[2]], dtype=img.dtype), img[1:,:,:] - img[:-1,:,:]], axis=0)
  # # gy ditto
  # gy = tf.concat([tf.zeros([img.shape[1],1,img.shape[2]], dtype=img.dtype), img[:,1:,:] - img[:,:-1,:]], axis=1)

  sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
  sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
  sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

  # Shape = 1 x height x width x 1.
  gx = tf.nn.conv2d(tf.expand_dims(img, -1), sobel_x_filter,
                            strides=[1, 1, 1, 1], padding='SAME')
  gy = tf.nn.conv2d(tf.expand_dims(img, -1), sobel_y_filter,
                            strides=[1, 1, 1, 1], padding='SAME')

  gx = tf.squeeze(gx, axis=3)
  gy = tf.squeeze(gy, axis=3)

  # Reduce max gx and gy across their 3 channels
  gx = tf.reduce_max(gx, reduction_indices=[2], keepdims=True)
  gy = tf.reduce_max(gy, reduction_indices=[2], keepdims=True)

  # img = tf.concat([img, gx, gy], axis=2)
  img = tf.concat([gx, gy], axis=2) # Only keep image gradients
  return img

def input_fn(inp_dataset, is_training=False, batch_size=50, buffer_size=1000000, prefetch_buffer_size=5000):
  # Returns an appropriate input function for training/evaluation.
  def sub_input_fn():
    dataset = inp_dataset

    # Shuffle if training
    if is_training:
      dataset = dataset.shuffle(buffer_size=buffer_size)

    # Make a float image
    dataset = dataset.map(lambda img, label: (tf.to_float(img), label), num_parallel_calls=4)

    if is_training:
      # Slightly randomize images.
      dataset = dataset.map(lambda img, label: (randomize_image(img), label), num_parallel_calls=4)


    # Get image gradient only, turning into 2 channels gx and gy
    # dataset = dataset.map(lambda img, label: (add_image_gradient(tf.to_float(img)), label), num_parallel_calls=4)

    # Bring down to 45x45 to 31x31
    dataset = dataset.map(lambda img, label: (tf.reshape(
        tf.image.central_crop(img, 0.6888888888), [15,15,3], name='ReshapeAfterCentalCrop')
      , label), num_parallel_calls=4)
    # dataset = dataset.map(lambda img, label: (tf.image.central_crop(img, 0.66666666666), label), num_parallel_calls=4)


    # Shuffle/Batch/prefetch should happen after size changes

    # Batch and repeat.
    if is_training:
      dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size) # Doesn't necessarily need to be same as shuffle.

    # Build iterator and return
    one_shot_iterator = dataset.make_one_shot_iterator()
    next_element = one_shot_iterator.get_next()

    # Return in a dict so the premade estimators can use it.
    return {"x": next_element[0]}, next_element[1]
  return sub_input_fn

def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string),
              "label": tf.FixedLenFeature((), tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  image = parsed_features['image']
  image = tf.decode_raw(image, tf.uint8, name='InitDecodeRaw')
  # image = tf.reshape(image,[45, 45, 3])
  # Note: Make sure input dataset isn't corrupted with different sized tensors (via say RGBA instead of RGB)
  image = tf.reshape(image,[21, 21, 3], name='InitReshape21x21x3')
  return image, parsed_features["label"]

def main(args):
  random.seed(100)

  # filenames = glob.glob('%s/*.tfrecords' % args.tfrecords_path)
  # train_filenames = filenames[:-2]
  # test_filenames = filenames[2:]
  # winsize22
 #  train_filenames = ['../datasets/tfrecords/winsize_22_color/wgm_1_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/swivel_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/sam2_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/speedchess1_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/carlsen_match2_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/output_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/john2_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/match1_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/john1_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/random1_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/gm_magnus_1_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/chess_beer_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/match2_ws22.tfrecords',
 # '../datasets/tfrecords/winsize_22_color/bro_1_ws22.tfrecords']
 #  test_filenames = ['../datasets/tfrecords/winsize_22_color/sam1_ws22.tfrecords']
  train_filenames = [
                     # '../datasets/tfrecords/winsize_10_color/wgm_1_ws10.tfrecords',
                     '../datasets/tfrecords/winsize_10_color/input_images_ws10.tfrecords',
                     # '../datasets/tfrecords/winsize_10_color/swivel_ws10.tfrecords',
                     '../datasets/tfrecords/winsize_10_color/sam2_ws10.tfrecords',
                     '../datasets/tfrecords/winsize_10_color/speedchess1_ws10.tfrecords',
                     # '../datasets/tfrecords/winsize_10_color/carlsen_match2_ws10.tfrecords',
                     # '../datasets/tfrecords/winsize_10_color/output_ws10.tfrecords',
                     '../datasets/tfrecords/winsize_10_color/john2_ws10.tfrecords',
                     # '../datasets/tfrecords/winsize_10_color/match1_ws10.tfrecords',
                     '../datasets/tfrecords/winsize_10_color/john1_ws10.tfrecords',
                     # '../datasets/tfrecords/winsize_10_color/random1_ws10.tfrecords',
                     # '../datasets/tfrecords/winsize_10_color/gm_magnus_1_ws10.tfrecords',
                     '../datasets/tfrecords/winsize_10_color/chess_beer_ws10.tfrecords',
                     # '../datasets/tfrecords/winsize_10_color/match2_ws10.tfrecords',
                     '../datasets/tfrecords/winsize_10_color/bro_1_ws10.tfrecords'
                     ]
  # train_filenames = ['../datasets/tfrecords/winsize_10_color/input_images_ws10.tfrecords',]
  # train_filenames = ['../datasets/tfrecords/winsize_10_color/sam2_ws10.tfrecords',]
  test_filenames = ['../datasets/tfrecords/winsize_10_color/sam1_ws10.tfrecords'] # Used for ultrasmall v3 96%
  # test_filenames = ['../datasets/tfrecords/winsize_10_color/chess_beer_ws10.tfrecords']


  with tf.Session() as sess:
    print("In session")  
    train_dataset = tf.data.TFRecordDataset(train_filenames)
    # Convert img byte str back to 21x21 np.uint8 array.
    train_dataset = train_dataset.map(_parse_function)

    test_dataset = tf.data.TFRecordDataset(test_filenames)
    test_dataset = test_dataset.map(_parse_function)

    # Build model.
    model_dir = './training_models/cnn_tfrecord_%s' % (args.run_name)

    ###
    # one_shot_iterator = train_dataset.make_one_shot_iterator()
    # next_element = one_shot_iterator.get_next()
    # a,b = next_element[0], next_element[1]
    # for i in range(100):
    #   x = sess.run(a)
    #   if x.shape != (21,21,3):
    #     print(i, x.shape, x.dtype)
    # exit()
    ###

    ###
    # a, b = input_fn(train_dataset, is_training=True)()
    # x = sess.run(a)['x']
    # print(x.shape)
    # print(x.dtype)
    # exit()
    ###

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
      print("\n\tTraining %d-%d/%d" % (i*train_steps,(i+1)*train_steps,args.steps*train_steps))
      ta = time()
      estimator.train(
        input_fn=input_fn(train_dataset, is_training=True, batch_size=args.batch_size),
        steps=train_steps)
      tb = time()

      # Evaluate
      print('\n\tEvaluating...')
      metrics = estimator.evaluate(input_fn=input_fn(test_dataset))
      accuracy_score = metrics["accuracy"]
      print("-- Test Accuracy: {0:f}\n".format(accuracy_score))
      tc = time()
      print("Train took %g, Evaluate took %g" % (tb-ta, tc-tb))

if __name__ == '__main__':
  parser = ArgumentParser()
  # parser.add_argument("-m", "--max_each", dest="max_count_each_entries", type=int,
  #                     help="Maximum count of good or bad tiles per each folder")
  parser.add_argument("--name", dest="run_name", required=True,
                      help="Name of model run name")
  parser.add_argument("--tfrecords_path", default='datasets/tfrecords/winsize10',
                      help="Folder to load tfrecord output")
  parser.add_argument("-s", "--steps", dest="steps", required=True, type=int,
                      help="Number of eval steps to run (x1000 train steps).")
  parser.add_argument("-bs", "--batch_size", type=int, default=50, help="Batch size.")
  parser.add_argument("-fs", "--filter_sizes", dest="filter_sizes", nargs='+', type=int,
                      default=[32, 64, 1024], help="CNN model filter sizes")


  args = parser.parse_args()
  print("Arguments passed: \n\t%s\n" % args)
  main(args)