from __future__ import print_function
from pylab import *
import numpy as np
import cv2
import sys


def main(filenames):
  for filename in filenames:
    print("Processing %s" % filename)
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Using opencv
    cv2.imshow('image %dx%d' % (img.shape[1],img.shape[0]),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
  if len(sys.argv) > 1:
    filenames = sys.argv[1:]
  else:
    filenames = ['input/6.jpg']
  main(filenames)
