from __future__ import print_function
from pylab import *
import numpy as np
import cv2
import sys
import IPython
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure

np.set_printoptions(suppress=True, precision=2) # Better printing of arrays
# def getHog


def main(filenames):
  for filename in filenames:
    print("Processing %s" % filename)
    img = cv2.imread(filename)
    if (img.size > 1000*1000):
      img = cv2.resize(img,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Edges
    # edges = cv2.Canny(gray,200,500,apertureSize = 3, L2gradient=False) # Better thresholds

    # # Gradients
    # sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    # sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    # grad_mag = np.sqrt(sobelx**2+sobely**2)
    # grad_phase = np.arctan2(sobely, sobelx)
    # grad_mag_norm = grad_mag + grad_mag[:].min()
    # grad_mag_norm = (grad_mag_norm / grad_mag_norm[:].max())

    # grad_phase_norm = (grad_phase + np.pi)/(2*np.pi) * grad_mag_norm
    # w,h,_ = grad_phase_norm.shape
    # grad_phase_norm[grad_mag_norm < 0.05] = 0

    fd, hog_image = hog(gray, orientations=32, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(gray, cmap=plt.cm.gray)
    ax1.grid('on')
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10.0))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.grid('on')
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')

    # pt = (139,304)
    # ax1.plot(pt[0], pt[1],'o')
    # ax2.plot(pt[0], pt[1],'o')

    plt.show()

    




    # Using opencv
    # cv2.imshow('image %dx%d' % (img.shape[1],img.shape[0]),img)
    # cv2.imshow('grad', grad_mag_norm)
    # cv2.imshow('phase', grad_phase_norm)
    # cv2.imshow('hist', h)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
  if len(sys.argv) > 1:
    filenames = sys.argv[1:]
  else:
    filenames = ['input/21.jpg']
  main(filenames)
