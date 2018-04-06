# coding=utf-8
import PIL.Image
import matplotlib.image as mpimg
import scipy.ndimage
import cv2 # For Sobel etc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays


featureA = tf.feature_column.numeric_column("x", shape=[11,11], dtype=tf.uint8)

estimator = tf.estimator.DNNClassifier(
  feature_columns=[featureA],
  hidden_units=[256, 32],
  n_classes=2,
  dropout=0.1,
  model_dir='./xcorner_model_7k',
  )


# Load pt_dataset.txt and generate the windowed tiles for all the good and bad
# points in folders dataset/good dataset/bad


def loadImage(filepath, doGrayscale=False):
    img_orig = PIL.Image.open(filepath)
    img_width, img_height = img_orig.size

    # Resize
    aspect_ratio = min(500.0/img_width, 500.0/img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width,new_height), resample=PIL.Image.BILINEAR)
    # if (doGrayscale):
    img_gray = img.convert('L') # grayscale
    img = np.array(img)
    img_gray = np.array(img_gray)
    
    return img, img_gray

import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if    os.path.isdir(path):
            pass
        else:
            raise

def main():
  input_data = 'pt_dataset.txt'

  results_folder = 'ml_xcorner_results'
  mkdir_p(results_folder)

  WINSIZE = 5

  DO_BINARIZATION = False
  DO_OPENING = False

  if (DO_BINARIZATION):
    raise('Error, must be grayscale if doing binarization.')

  count_good = 0
  count_bad = 0
  
  # save all points to a file
  with open('pt_dataset.txt', 'r') as f:
    lines = [x.strip() for x in f.readlines()]

  n = len(lines)/5
  # n = 1

  def input_fn_predict(img_data): # returns x, None
    def ret_func():
      dataset = tf.data.Dataset.from_tensor_slices(
          {
          'x':img_data
          }
        )
      # return dataset.make_one_shot_iterator().get_next(), tf.one_hot(labels,2,dtype=tf.int32)
      dataset = dataset.batch(25)
      iterator = dataset.make_one_shot_iterator()
      k = iterator.get_next()
      return k, None
    return ret_func


  for i in range(n):
    print("On %d/%d" % (i+1, n))
    filename = lines[i*5]
    s0 = lines[i*5+1].split()
    s1 = lines[i*5+2].split()
    s2 = lines[i*5+3].split()
    s3 = lines[i*5+4].split()
    good_pts = np.array([s1, s0], dtype=np.int).T
    bad_pts = np.array([s3, s2], dtype=np.int).T

    img_filepath = 'input/%s.png' % filename
    if not os.path.exists(img_filepath):
      img_filepath = 'input/%s.jpg' % filename
    img, img_gray = loadImage(img_filepath)

    kernel = np.ones((3,3),np.uint8)

    tiles = []
    all_pts = []

    # Good points
    for i in range(good_pts.shape[0]):
      pt = good_pts[i,:]
      if (np.any(pt <= WINSIZE) or np.any(pt >= np.array(img_gray.shape[:2]) - WINSIZE)):
        # print("Skipping point %s" % pt)
        continue
      else:
        tile = img_gray[pt[0]-WINSIZE:pt[0]+WINSIZE+1, pt[1]-WINSIZE:pt[1]+WINSIZE+1]
        if DO_BINARIZATION:
          tile = cv2.adaptiveThreshold(tile,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        
        if DO_OPENING:
          tile = cv2.morphologyEx(tile, cv2.MORPH_OPEN, kernel)

        tiles.append(tile)
        all_pts.append(pt)
        count_good += 1

    # Bad points
    for i in range(bad_pts.shape[0]):
      pt = bad_pts[i,:]
      if (np.any(pt <= WINSIZE) or np.any(pt >= np.array(img_gray.shape[:2]) - WINSIZE)):
        continue
      else:
        tile = img_gray[pt[0]-WINSIZE:pt[0]+WINSIZE+1, pt[1]-WINSIZE:pt[1]+WINSIZE+1]
        if DO_BINARIZATION:
          tile = cv2.adaptiveThreshold(tile,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        if DO_OPENING:
          tile = cv2.morphologyEx(tile, cv2.MORPH_OPEN, kernel)
        

        tiles.append(tile)
        all_pts.append(pt)
        count_bad += 1

    tiles = np.array(tiles)
    all_pts = np.array(all_pts)

    # Do prediction  
    import time
    a = time.time()
    predictions = estimator.predict(input_fn=input_fn_predict(tiles))

    for i, prediction in enumerate(predictions):
      c = prediction['probabilities'].argmax()
      pt = all_pts[i]
      if (c == 1):
        cv2.circle(img, tuple(pt[::-1]), 4, (0,255,0), -1)
      else:
        cv2.circle(img, tuple(pt[::-1]), 3, (255,0,0), -1)
    b = time.time()
    print("predict() took %.2f ms" % ((b-a)*1e3))


    im = PIL.Image.fromarray(img).convert('RGB')
    im.save('%s/%s_xcorner_7k.png' % (results_folder, filename))

  print ("Finished %d good and %d bad tiles" % (count_good, count_bad))



if __name__ == '__main__':
  main()



