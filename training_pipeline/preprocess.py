# Methods to load datasets from folders, preprocess them,
# and build input functions for the estimators.
import tensorflow as tf
import glob
import numpy as np
from tensorflow.contrib.data import Dataset
import random

def loadDatapaths(parent_folder, max_count_each_entries=None, do_shuffle=True):
  # Builds and returns a dataset of images from the
  # good/ and bad/ subfolders of the parent folder.
  folder_good = '%s/good' % parent_folder
  folder_bad = '%s/bad' % parent_folder

  filepaths_good = glob.glob("%s/*.png" % folder_good)
  filepaths_bad = glob.glob("%s/*.png" % folder_bad)

  # Use only up to max_count_each_entries of each for training equally.
  if max_count_each_entries:
    filepaths_good = filepaths_good[:max_count_each_entries]
    filepaths_bad = filepaths_bad[:max_count_each_entries]

  N_good, N_bad = len(filepaths_good), len(filepaths_bad)

  # Set up labels
  labels = np.array([1]*N_good + [0]*N_bad, dtype=np.float64)

  # Shuffle all entries keeping labels and paths together.
  entries = zip(filepaths_good + filepaths_bad, labels)
  if do_shuffle:
    random.shuffle(entries)
  # Separate back into imgs / labels and return.
  imgs, labels = zip(*entries)
  return imgs, labels

def buildDataset(img_paths, labels, train_test_split_percentage=0.8):
  # Split into training and test
  split = int(len(img_paths) * train_test_split_percentage)
  tr_imgs = tf.constant(img_paths[:split])
  tr_labels = tf.constant(labels[:split])
  val_imgs = tf.constant(img_paths[split:])
  val_labels = tf.constant(labels[split:])

  return tr_imgs, tr_labels, val_imgs, val_labels

def input_parser(img_path, label):
  # Read the img from file.
  img_file = tf.read_file(img_path)
  img_decoded = tf.image.decode_image(img_file, channels=1)

  return img_decoded, label

def randomize_image(img, contrast_range=[0.2,1.8], brightness_max=63):
  # Apply random flips/rotations and contrast/brightness changes to image
  img = tf.image.random_flip_left_right(img)
  img = tf.image.random_flip_up_down(img)
  img = tf.image.rot90(img, k=np.random.randint(4))
  img = tf.image.random_contrast(img, lower=contrast_range[0], upper=contrast_range[1])
  img = tf.image.random_brightness(img, max_delta=brightness_max)
  return img

def preprocessor(dataset, batch_size, dataset_length=None, is_training=False):
  if is_training and dataset_length:
    # Shuffle dataset.
    dataset = dataset.shuffle(dataset_length*2)

  # Load images from image paths.
  dataset = dataset.map(input_parser)

  if is_training:
    # Slightly randomize images.
    dataset = dataset.map(lambda img, label: (randomize_image(img), label))

  # Zero mean and unit normalize images, float image output.
  dataset = dataset.map(lambda img, label: (tf.image.per_image_standardization(img), label))

  # Batch and repeat.
  dataset = dataset.batch(batch_size)
  if is_training:
    dataset = dataset.repeat()

  return dataset

def input_fn(imgs, labels, dataset_length=None, is_training=True, batch_size=50):
  # Returns an appropriate input function for training/evaluation.
  def sub_input_fn():
    dataset = Dataset.from_tensor_slices((imgs, labels))
    # Pre-process dataset into correct form/batching/shuffle etc.
    dataset = preprocessor(dataset, batch_size, dataset_length, is_training)

    # Build iterator and return
    one_shot_iterator = dataset.make_one_shot_iterator()
    next_element = one_shot_iterator.get_next()

    # Return in a dict so the premade estimators can use it.
    return {"x": next_element[0]}, next_element[1]
  return sub_input_fn