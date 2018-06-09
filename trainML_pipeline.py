# training pipeline loading images from filepaths
import tensorflow as tf
import glob
import numpy as np
from tensorflow.contrib.data import Dataset, Iterator
import random
random.seed(100)


dataset_folder = 'dataset_gray_10'
folder_good = '%s/good' % dataset_folder
folder_bad = '%s/bad' % dataset_folder


filepaths_good = glob.glob("%s/*.png" % folder_good)
filepaths_bad = glob.glob("%s/*.png" % folder_bad)

# # Shuffle order
# random.shuffle(filepaths_good)
# random.shuffle(filepaths_bad)

# TODO : Temporarily only using 6000 of each for training equally.
# filepaths_good = filepaths_good[:6000]
# filepaths_bad = filepaths_bad[:6000]

N_good, N_bad = len(filepaths_good), len(filepaths_bad)
print(N_good, N_bad)

# Set up labels
labels = np.zeros((N_good+N_bad))
labels[:N_good] = 1


# Shuffle all entries keeping labels and paths together, then separate back into f_imgs/labels
entries = zip(filepaths_good + filepaths_bad, labels)
random.shuffle(entries)
f_imgs, f_labels = zip(*entries)


# Split into training and test
split = int(len(entries) * 0.8) # 5000
tr_imgs = tf.constant(f_imgs[:split])
tr_labels = tf.constant(f_labels[:split])
val_imgs = tf.constant(f_imgs[split:])
val_labels = tf.constant(f_labels[split:])

# tr_data = Dataset.from_tensor_slices((tr_imgs, tr_labels))
# val_data = Dataset.from_tensor_slices((val_imgs, val_labels))


# let's assume we have two classes in our dataset
NUM_CLASSES = 2

def input_parser(img_path, label):
  # convert the label to one-hot encoding
  # one_hot = tf.one_hot(label, NUM_CLASSES)

  # read the img from file
  img_file = tf.read_file(img_path)
  img_decoded = tf.image.decode_image(img_file, channels=1)

  # return img_decoded, one_hot
  return img_decoded, label

# # Map the actual images and one-hot labels for the data
# tr_data = tr_data.map(input_parser)
# val_data = val_data.map(input_parser)

# # Batch and repeat.
# batch_size = 20
# tr_data = tr_data.batch(batch_size).repeat()
# val_data = val_data.batch(batch_size).repeat()

# create TensorFlow Iterator object
# iterator = Iterator.from_structure(tr_data.output_types,
#                                    tr_data.output_shapes)
# next_element = iterator.get_next()

# # create two initialization ops to switch between the datasets
# training_init_op = iterator.make_initializer(tr_data)
# validation_init_op = iterator.make_initializer(val_data)

# def input_fn(feats, labs, do_shuffle=True):
#   def return_fn():
#     dataset = tf.data.Dataset.from_tensor_slices(
#         {
#         'x':feats, 
#         'label':labs
#         }
#       )
#     if (do_shuffle):
#       dataset = dataset.shuffle(len(feats)*2)
#     dataset = dataset.batch(25).repeat()

#     dataset_one_shot_iterator = dataset.make_one_shot_iterator().get_next()
#     return dataset_one_shot_iterator, dataset_one_shot_iterator['label']
#   return return_fn

def randomize_image(img):
  img = tf.image.random_flip_left_right(img)
  img = tf.image.random_flip_up_down(img)
  img = tf.image.random_contrast(img, lower=0.4, upper=0.6)
  img = tf.image.random_brightness(img, max_delta=0.2)
  return img

def train_input_fn():
  tr_data = Dataset.from_tensor_slices((tr_imgs, tr_labels))
  tr_data = tr_data.map(input_parser)
  tr_data = tr_data.shuffle(split*2)

  # Batch and repeat.
  batch_size = 50
  tr_data = tr_data.batch(batch_size).repeat()

  one_shot_iterator = tr_data.make_one_shot_iterator()
  next_element = one_shot_iterator.get_next()

  return {"x": randomize_image(next_element[0])}, next_element[1]

def validate_input_fn():
  val_data = Dataset.from_tensor_slices((val_imgs, val_labels))
  val_data = val_data.map(input_parser)
  # Batch
  batch_size = 50
  val_data = val_data.batch(batch_size) # no repeat
  one_shot_iterator = val_data.make_one_shot_iterator()
  next_element = one_shot_iterator.get_next()

  return {"x": next_element[0]}, next_element[1]

feature_img = tf.feature_column.numeric_column("x", shape=[21,21], dtype=tf.uint8)


units = [512,256]
estimator = tf.estimator.DNNClassifier(
  feature_columns=[feature_img],
  hidden_units=units,
  n_classes=2,
  # dropout=0.1,
  model_dir='./training_models/v_win%s_%s' % (dataset_folder[dataset_folder.rfind('_')+1:], "_".join(map(str,units))),
  # model_dir='./training_models/lin_win%s' % (dataset_folder[dataset_folder.rfind('_')+1:]),
  # optimizer=tf.train.ProximalAdagradOptimizer(
  #   learning_rate=0.001,
  #   l1_regularization_strength=0.001
  # )
  )

# with tf.Session() as sess:
#   print(sess.run(train_input_fn()))

############################
# Train
n = 1000
k = 1000
for i in range(n):
  print("Training %d-%d/%d" % (i*k,(i+1)*k,n*k))
  estimator.train(input_fn=train_input_fn, steps=k)

  accuracy_score = estimator.evaluate(input_fn=validate_input_fn)["accuracy"]
  print("-- Test Accuracy: {0:f}\n".format(accuracy_score))


############################
# Validate
predictions = estimator.predict(input_fn=validate_input_fn)

test_labels = np.array(f_labels[split:])

count_good = 0
for i,(prediction,true_answer) in enumerate(zip(predictions, test_labels.astype(np.int))):
  if prediction['probabilities'].argmax() != true_answer:
    # print("Failure on %d: %s" % (i, prediction['probabilities']))
    continue
  else:
    count_good += 1

success_rate = float(count_good) / len(test_labels)
print("Total %d/%d right ~ %.2f%% success rate" % (count_good, len(test_labels), 100*success_rate))

####
# Save model
# export_dir = 'ml/model/002'
# def serving_input_receiver_fn():
#   """Build the serving inputs."""
#   # The outer dimension (None) allows us to batch up inputs for
#   # efficiency. However, it also means that if we want a prediction
#   # for a single instance, we'll need to wrap it in an outer list.
#   inputs = {"x": tf.placeholder(shape=[None, 11, 11], dtype=tf.uint8)}
#   return tf.estimator.export.ServingInputReceiver(inputs, inputs)

# estimator.export_savedmodel(export_dir, serving_input_receiver_fn)

# with tf.Session() as sess:

#     # initialize the iterator on the training data
#     sess.run(training_init_op)

#     # get each element of the training dataset until the end is reached
#     while True:
#         try:
#             elem = sess.run(next_element) # batch_size x 21 x 21 x 1
#             print(elem[0].shape)
#         except tf.errors.OutOfRangeError:
#             print("End of training dataset.")
#             break

#     # initialize the iterator on the validation data
#     sess.run(validation_init_op)

#     # get each element of the validation dataset until the end is reached
#     # while True:
#     #     try:
#     #         elem = sess.run(next_element)
#     #         print(elem)
#     #     except tf.errors.OutOfRangeError:
#     #         print("End of training dataset.")
#     #         break