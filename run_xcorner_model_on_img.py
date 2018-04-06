import numpy as np
import tensorflow as tf
import PIL.Image
np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays


featureA = tf.feature_column.numeric_column("x", shape=[11,11], dtype=tf.uint8)

estimator = tf.estimator.DNNClassifier(
  feature_columns=[featureA],
  hidden_units=[256, 32],
  n_classes=2,
  dropout=0.1,
  model_dir='./xcorner_model2',
  )


# filepath = 'input/img_01.jpg'
filepath = 'testB.jpg'
img = PIL.Image.open(filepath).convert('L') # Grayscale full image with chessboard
img_width, img_height = img.size
aspect_ratio = min(200.0/img_width, 200.0/img_height)
new_width, new_height = ((np.array(img.size) * aspect_ratio)).astype(int)
img = img.resize((new_width,new_height), resample=PIL.Image.BILINEAR)
img = np.array(img)

winsize = 5

new_size = [img.shape[0]-2*winsize, img.shape[1]-2*winsize]
print("out_size : " , new_size)

tiles = []

for r in range(winsize+1,img.shape[0]-winsize):
  for c in range(winsize+1,img.shape[1]-winsize):
    win = img[r-winsize:r+winsize+1,c-winsize:c+winsize+1]
    tiles.append(win)

img_features = np.array(tiles)
print(img_features.shape)

# filepath = 'dataset_gray_5/bad/img_01_002.png'
# img = PIL.Image.open(filepath)
# img_features = np.array([np.array(img)])

def input_fn_predict(): # returns x, None
  dataset = tf.data.Dataset.from_tensor_slices(
      {
      'x':img_features
      }
    )
  # return dataset.make_one_shot_iterator().get_next(), tf.one_hot(labels,2,dtype=tf.int32)
  dataset = dataset.batch(25)
  iterator = dataset.make_one_shot_iterator()
  k = iterator.get_next()
  return k, None

# Do prediction  
predictions = estimator.predict(input_fn=input_fn_predict)

heatmap = np.zeros(img_features.shape[0])
for i,prediction in enumerate(predictions):
  if (i % 1000 == 0):
    print ('%d / %d ' % (i, len(img_features)))
  if (prediction['probabilities'].argmax() == 1):
    heatmap[i] = prediction['probabilities'][1] # Probability is saddle
  # heatmap[i] = prediction['probabilities'].argmax() # Final answer

np.save('heatmap', heatmap)
print(heatmap)