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

# Reverse x and y in positions/*_all.txt once

def main():
  # all_pts
  filenames = glob.glob('positions/img_*_all.txt')
  filenames = sorted(filenames)
  n = len(filenames)

  for i in range(n):
      filename = filenames[i]
      print ("Processing %d/%d : %s" % (i+1,n,filename))

      all_pts = np.loadtxt(filename)
      all_pts = all_pts[:,[1,0]]
      np.savetxt(filename, all_pts)

if __name__ == '__main__':
  main()



