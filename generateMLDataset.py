# coding=utf-8
import PIL.Image
import matplotlib.image as mpimg
import scipy.ndimage
import cv2 # For Sobel etc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays

# Load pt_dataset.txt and generate the windowed tiles for all the good and bad
# points in folders dataset/good dataset/bad


def loadImage(filepath, doGrayscale=False):
    img_orig = PIL.Image.open(filepath)
    img_width, img_height = img_orig.size

    # Resize
    aspect_ratio = min(500.0/img_width, 500.0/img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width,new_height), resample=PIL.Image.BILINEAR)
    if (doGrayscale):
      img = img.convert('L') # grayscale
    img = np.array(img)
    
    return img

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

  WINSIZE = 5
  dataset_folder = 'dataset_gray_%d' % WINSIZE

  DO_GRAYSCALE = True
  DO_BINARIZATION = False
  DO_OPENING = False

  if (DO_BINARIZATION and not DO_GRAYSCALE):
    raise('Error, must be grayscale if doing binarization.')

  count_good = 0
  count_bad = 0

  good_features = []
  good_labels = []
  bad_features = []
  bad_labels = []

  # save all points to a file
  with open('pt_dataset2.txt', 'r') as f:
    lines = [x.strip() for x in f.readlines()]
    n = len(lines)/5
    # n = 1
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
      if not os.path.exists(img_filepath):
        img_filepath = 'input_yt/%s.jpg' % filename
      if not os.path.exists(img_filepath):
        img_filepath = 'input_yt/%s.png' % filename
      img = loadImage(img_filepath, DO_GRAYSCALE)

      kernel = np.ones((3,3),np.uint8)

      # Good points
      for i in range(good_pts.shape[0]):
        pt = good_pts[i,:]
        if (np.any(pt <= WINSIZE) or np.any(pt >= np.array(img.shape[:2]) - WINSIZE)):
          continue
        else:
          tile = img[pt[0]-WINSIZE:pt[0]+WINSIZE+1, pt[1]-WINSIZE:pt[1]+WINSIZE+1]

          if DO_BINARIZATION:
            tile = cv2.adaptiveThreshold(tile,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
          
          if DO_OPENING:
            tile = cv2.morphologyEx(tile, cv2.MORPH_OPEN, kernel)


          good_features.append(tile)
          good_labels.append(1)

          count_good += 1

      # Bad points
      for i in range(bad_pts.shape[0]):
        pt = bad_pts[i,:]
        if (np.any(pt <= WINSIZE) or np.any(pt >= np.array(img.shape[:2]) - WINSIZE)):
          continue
        else:
          tile = img[pt[0]-WINSIZE:pt[0]+WINSIZE+1, pt[1]-WINSIZE:pt[1]+WINSIZE+1]
          if DO_BINARIZATION:
            tile = cv2.adaptiveThreshold(tile,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
          if DO_OPENING:
            tile = cv2.morphologyEx(tile, cv2.MORPH_OPEN, kernel)

          bad_features.append(tile)
          bad_labels.append(0)

          count_bad += 1

  features = np.array(good_features + bad_features)
  print(features.shape)
  labels = np.array(good_labels + bad_labels, dtype=np.float32)
  print(labels.shape)

  np.savez('dataset2_%d' % WINSIZE, features=features, labels=labels)
  # Example to use: print(np.load('dataset_5.npz')['features'])
  
  print ("Finished %d good and %d bad tiles" % (count_good, count_bad))



if __name__ == '__main__':
  main()



