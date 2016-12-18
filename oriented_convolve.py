from __future__ import print_function
from pylab import *
import numpy as np
import scipy.ndimage
import cv2
import sys
from board_detect import *

def getCornerNormals(corners):
  # 4x2 array, rows are each point, columns are x and y
  dirs = np.zeros([4,2])

  # Side lengths of rectangular contour
  dirs[0,:] = (corners[1,:] - corners[0,:]) / np.sqrt(np.sum((corners[1,:] - corners[0,:])**2))
  dirs[1,:] = (corners[2,:] - corners[1,:]) / np.sqrt(np.sum((corners[2,:] - corners[1,:])**2))
  dirs[2,:] = (corners[3,:] - corners[2,:]) / np.sqrt(np.sum((corners[3,:] - corners[2,:])**2))
  dirs[3,:] = (corners[0,:] - corners[3,:]) / np.sqrt(np.sum((corners[0,:] - corners[3,:])**2))

  # Rotate 90 deg to get normal vector via [-y,x]
  normals = -np.hstack([-dirs[:,[1]], dirs[:,[0]]])
  # normals = dirs
  return normals, dirs


def main(filenames):
  for filename in filenames:
    print("Processing %s" % filename)
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    contours, chosen_tile_idx, edges = findPotentialTiles(img)
    if not len(contours):
      return
    drawPotentialTiles(img, contours, chosen_tile_idx)

    tile_corners = getChosenTile(contours, chosen_tile_idx)

    hough_corners, corner_hough_lines, edges_roi = refineTile(img, edges, contours, chosen_tile_idx)
    drawBestHoughLines(img, hough_corners, corner_hough_lines)

    corner_normals, _ = getCornerNormals(hough_corners)

    # print(hough_corners)
    # print(corner_normals)

    # Corner kernel
    # kX = np.array([[3,10,3],[0,0,0],[-3,-10,-3]])
    # kY = np.array([[3,10,3],[0,0,0],[-3,-10,-3]]).T

    # Gradient of the combination of scharr X and Y operators
    # kX = np.array([[9,9,9],[0,0,0],[-9,-9,-9]])
    # kY = np.array([[9,9,9],[0,0,0],[-9,-9,-9]]).T

    # kernel = kX * corner_normals[0,0] + kY * corner_normals[0,1]
    # print(kernel)

    # responseA = scipy.ndimage.filters.convolve(edges.astype(float), kernel, mode='constant')

    # Gradients
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    grad_mag = np.sqrt(sobelx**2+sobely**2)

    responseA = sobelx*corner_normals[0,0] + sobely*corner_normals[0,1]
    responseB = sobelx*corner_normals[1,0] + sobely*corner_normals[1,1]



    # # Normalize by absolute 
    responseA = abs(responseA)
    responseB = abs(responseB)
    # # responseA /= responseA.max()

    # Normalize Response to 0-1 range
    # responseA = -responseA
    a,b = responseA.min(), responseA.max()
    responseA = ((responseA-a)/(b-a))

    a,b = responseB.min(), responseB.max()
    responseB = ((responseB-a)/(b-a))
    
    # responseA[grad_mag<grad_mag.mean()*2] = 0
    # responseB[grad_mag<grad_mag.mean()*2] = 0
    # responseA[responseA<0.3] = 0

    # print(responseA.min())
    # print(responseA.max())
    # # responseA[responseA<200] = 0


    for i in range(1):
      ctr_line = (hough_corners[i,:] + hough_corners[(i+1)%4,:])/2
      a0 = (ctr_line).astype(int)
      a1 = (ctr_line + corner_normals[i,:]*20).astype(int)
      cv2.line(img,tuple(a0),tuple(a1), (0,0,255),2)




    # Using opencv
    cv2.imshow('image %dx%d' % (img.shape[1],img.shape[0]),img)
    cv2.imshow('convA', responseA)
    cv2.imshow('convB', responseB)
    # cv2.imshow('edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
  if len(sys.argv) > 1:
    filenames = sys.argv[1:]
  else:
    filenames = ['input2/27.jpg']
  main(filenames)
