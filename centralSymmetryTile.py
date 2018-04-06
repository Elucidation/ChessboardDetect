# coding=utf-8
import PIL.Image
import matplotlib.image as mpimg
import scipy.ndimage
import cv2 # For Sobel etc
import glob
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import os
np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays


def getRingIndices(radius):
  # Bottom
  row1 = np.ones(radius*2+1, dtype=int)*radius
  col1 = np.arange(radius*2+1)-radius

  # Right
  row2 = -np.arange(1,radius*2+1)+radius
  col2 = np.ones(radius*2, dtype=int)*radius

  # Top
  row3 = -np.ones(radius*2, dtype=int)*radius
  col3 = -np.arange(1,radius*2+1)+radius

  # Left
  row4 = np.arange(1,radius*2+1-1)-radius
  col4 = -np.ones(radius*2-1, dtype=int)*radius

  rows = np.hstack([row1, row2, row3, row4])
  cols = np.hstack([col1, col2, col3, col4])
  return (rows,cols)

def countSteps(ring):
  # Build a big ring so we can handle circular edges
  bigring = np.hstack([ring,ring,ring])
  n = len(ring)
  # Go through middle portion of ring
  count = 0
  for i in (np.arange(n) + n):
    if (bigring[i] != bigring[i-1] and (bigring[i-1] == bigring[i-2]) and (bigring[i] == bigring[i+1])):
      count += 1
  return count



# Load a tile image and check the central symmetry around a ring
def main():
  bad_tile_filepaths = sorted(glob.glob('dataset_binary_5/bad/img_*.png'))
  good_tile_filepaths = sorted(glob.glob('dataset_binary_5/good/img_*.png'))

  # shuffle(bad_tile_filepaths)
  # shuffle(good_tile_filepaths)
  
  # Setup
  tile_radius = (PIL.Image.open(good_tile_filepaths[0]).size[0]-1)/2 #(img.shape[0]-1)/2
  radius = 5

  # filepath = 'dataset_binary_5/bad/img_01_008.png'
  # plt.figure(figsize=(20,20))
  # plt.subplot(121)
  # plt.title('False Positives')
  rows, cols = getRingIndices(radius)
  # Center in tile
  rows += tile_radius
  cols += tile_radius

  # for i in range(20):
  #   filepath = bad_tile_filepaths[i]
  #   img = PIL.Image.open(filepath).convert('L')
  #   img = np.array(img)
  #   # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
  #   ring = img[rows,cols]
  #   plt.plot(ring + i*255*2, '.-')
  #   plt.plot([0,len(ring)-1], np.ones(2) + 127 + i*255*2, 'k:', alpha=0.2)
  #   plt.text(0, i*255*2, countSteps(ring))

  # # Good tiles
  # plt.subplot(122)
  # plt.title('True Positives')
  # for i in range(20):
  #   filepath = good_tile_filepaths[i]
  #   img = PIL.Image.open(filepath).convert('L')
  #   img = np.array(img)
  #   # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
  #   ring = img[rows,cols]
  #   plt.plot(ring + i*255*2, '.-')
  #   plt.plot([0,len(ring)-1], np.ones(2) + 127 + i*255*2, 'k:', alpha=0.2)
  #   plt.text(0, i*255*2, countSteps(ring))

  # plt.show()

  good_steps = []
  bad_steps = []
  for i in range(len(bad_tile_filepaths)):
    filepath = bad_tile_filepaths[i]
    img = PIL.Image.open(filepath).convert('L')
    img = np.array(img)
    ring = img[rows,cols]
    steps = countSteps(ring)
    bad_steps.append(steps)
  for i in range(len(good_tile_filepaths)):
    filepath = good_tile_filepaths[i]
    img = PIL.Image.open(filepath).convert('L')
    img = np.array(img)
    ring = img[rows,cols]
    steps = countSteps(ring)
    good_steps.append(steps)

  # print(good_steps)
  # print(bad_steps)

  plt.subplot(121)
  plt.hist(bad_steps)
  plt.title('False Positives')
  plt.subplot(122)
  plt.hist(good_steps)
  plt.title('True Positives')
  plt.show()





if __name__ == '__main__':
  main()



