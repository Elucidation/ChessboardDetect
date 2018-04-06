import numpy as np
import tensorflow as tf
np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

with np.load('dataset2_5.npz') as np_dataset:
  # Get equal good/bad
  features = np_dataset['features']#[:(6227*2)]
  labels = np_dataset['labels']#[:(6227*2)]


# Shuffle data so good/bad are mixed
shuffle_order = np.arange(len(labels))
np.random.shuffle(shuffle_order)
features = features[shuffle_order,:,:]
labels = labels[shuffle_order]

# Split into train and test datasets
split = np.int(len(labels) * 0.8)
train_features = features[:split]
train_labels = labels[:split]

test_features = features[split:]
test_labels = labels[split:]


# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

# dataset = tf.data.Dataset.from_tensor_slices({'features':features, 'labels':labels}).repeat()
# Todo add placeholders to avoid repeat copies of data
# dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# iterator = tfe.Iterator(dataset)
# for i,value in enumerate(iterator):
#   print(i)
#   print(value)
#   if (i > 10):
#     break;

def input_fn(feats, labs, do_shuffle=True):
  def return_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        {
        'x':feats, 
        'label':labs
        }
      )
    if (do_shuffle):
      dataset = dataset.shuffle(len(feats)*2)
    dataset = dataset.batch(25).repeat()

    k = dataset.make_one_shot_iterator().get_next()
    return k, k['label']
  return return_fn

featureA = tf.feature_column.numeric_column("x", shape=[11,11], dtype=tf.uint8)

estimator = tf.estimator.DNNClassifier(
  feature_columns=[featureA],
  hidden_units=[256, 32],
  n_classes=2,
  dropout=0.1,
  model_dir='./xcorner_model_all',
  )

n = 20
for i in range(20):
  print("Training %d-%d/%d" % (i*1000,(i+1)*1000,n*1000))
  estimator.train(input_fn=input_fn(train_features, train_labels), steps=1000)

  print("Evaluation #%d" % (i+1))
  metrics = estimator.evaluate(input_fn=input_fn(test_features, test_labels), steps=1000)

def input_fn_predict(): # returns x, None
  dataset = tf.data.Dataset.from_tensor_slices(
      {
      'x':test_features
      }
    )
  # return dataset.make_one_shot_iterator().get_next(), tf.one_hot(labels,2,dtype=tf.int32)
  dataset = dataset.batch(25)
  iterator = dataset.make_one_shot_iterator()
  k = iterator.get_next()
  return k, None
  
predictions = estimator.predict(input_fn=input_fn_predict)

count_good = 0
for i,(prediction,true_answer) in enumerate(zip(predictions, test_labels.astype(np.int))):
  if prediction['probabilities'].argmax() != true_answer:
    print("Failure on %d: %s" % (i, prediction['probabilities']))
  else:
    count_good += 1

success_rate = float(count_good) / len(test_labels)
print("Total %d/%d right ~ %.2f%% success rate" % (count_good, len(test_labels), success_rate))