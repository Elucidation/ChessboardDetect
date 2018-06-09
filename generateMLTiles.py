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
  input_data = 'pt_dataset2.txt'

  WINSIZE = 10
  dataset_folder = 'dataset_gray_%d' % WINSIZE
  folder_good = '%s/good' % dataset_folder
  folder_bad = '%s/bad' % dataset_folder
  mkdir_p(folder_good)
  mkdir_p(folder_bad)

  DO_BINARIZATION = False
  DO_OPENING = False
  DO_GRAYSCALE = True

  if (DO_BINARIZATION and not DO_GRAYSCALE):
    raise('Error, must be grayscale if doing binarization.')

  count_good = 0
  count_bad = 0
  
  # save all points to a file
  with open(input_data, 'r') as f:
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
          # print("Skipping point %s" % pt)
          continue
        else:
          tile = img[pt[0]-WINSIZE:pt[0]+WINSIZE+1, pt[1]-WINSIZE:pt[1]+WINSIZE+1]
          # print(tile)
          out_filename = '%s/%s_%03d.png' % (folder_good, filename, i)
          if DO_BINARIZATION:
            tile = cv2.adaptiveThreshold(tile,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
          
          if DO_OPENING:
            tile = cv2.morphologyEx(tile, cv2.MORPH_OPEN, kernel)

          if DO_GRAYSCALE:
            im = PIL.Image.fromarray(tile).convert('L')
          else:
            im = PIL.Image.fromarray(tile).convert('RGB')
          im.save(out_filename)
          count_good += 1

      # Bad points
      for i in range(bad_pts.shape[0]):
        pt = bad_pts[i,:]
        if (np.any(pt <= WINSIZE) or np.any(pt >= np.array(img.shape[:2]) - WINSIZE)):
          # print("Skipping point %s" % pt)
          continue
        else:
          tile = img[pt[0]-WINSIZE:pt[0]+WINSIZE+1, pt[1]-WINSIZE:pt[1]+WINSIZE+1]
          out_filename = '%s/%s_%03d.png' % (folder_bad, filename, i)
          if DO_BINARIZATION:
            tile = cv2.adaptiveThreshold(tile,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
          if DO_OPENING:
            tile = cv2.morphologyEx(tile, cv2.MORPH_OPEN, kernel)

          if DO_GRAYSCALE:
            im = PIL.Image.fromarray(tile).convert('L')
          else:
            im = PIL.Image.fromarray(tile).convert('RGB')
          im.save(out_filename)
          count_bad += 1
  print ("Finished %d good and %d bad tiles" % (count_good, count_bad))



if __name__ == '__main__':
  main()



