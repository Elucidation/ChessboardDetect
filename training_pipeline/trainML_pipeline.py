# Training pipeline loading images from filepaths.
import tensorflow as tf
import numpy as np
import random
import preprocess
random.seed(100)

# Load dataset.
winsize=10
dataset_folder = '../datasets/dataset_gray_%d' % winsize
f_imgs, f_labels = preprocess.loadDatapaths(dataset_folder, max_count_each_entries=6000)
tr_imgs, tr_labels, val_imgs, val_labels = preprocess.buildDataset(f_imgs, f_labels)
dataset_length = len(f_imgs)
print("Loaded dataset '%s' : %d entries" % (dataset_folder, dataset_length))

# Build model.
feature_img = tf.feature_column.numeric_column("x", shape=[21,21], dtype=tf.uint8)
units = [512,256,128]
model_dir = './training_models/v_win%s_%s' % (winsize, "_".join(map(str,units)))
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
steps = 1000
train_steps = 1000 
for i in range(steps):
  # Test
  print("Training %d-%d/%d" % (i*train_steps,(i+1)*train_steps,steps*train_steps))
  estimator.train(input_fn=preprocess.input_fn(tr_imgs, tr_labels, dataset_length), steps=train_steps)

  # Evaluate
  metrics = estimator.evaluate(input_fn=preprocess.input_fn(val_imgs, val_labels, is_training=False))
  accuracy_score = metrics["accuracy"]
  print("-- Test Accuracy: {0:f}\n".format(accuracy_score))


############################
# Validate
predictions = estimator.predict(input_fn=preprocess.input_fn(val_imgs, val_labels, is_training=False))

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