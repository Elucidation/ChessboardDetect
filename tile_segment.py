# Image segmentation
# Given rectified image with known tile boundaries
# Segment image into background (black/white tiles?)
# and dark or light pieces
#
# Some options include K-means clustering, watershed segmentation, texture segmentation, perhaps a combination

import PIL.Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import itertools
import os
from skimage import color
from sklearn.cluster import KMeans
from skimage import exposure
np.set_printoptions(precision=2, linewidth=400, suppress=True) # Better printing of arrays

def getIdealCorners(tile_res, tile_buffer):
  hlines = vlines = (np.arange(9)+tile_buffer)*tile_res
  return np.array(list(itertools.product(hlines, vlines)))

def getIdealCheckerboardPattern(tile_res, tile_buffer):
  side_len = tile_res*(8+2*tile_buffer)
  quadOne = np.ones([tile_res,tile_res], dtype=np.uint8)
  quadZero = np.zeros([tile_res,tile_res], dtype=np.uint8)
  kernel = np.vstack([np.hstack([quadOne,quadZero]), np.hstack([quadZero,quadOne])])
  kernel = np.tile(kernel,(4,4)) # Becomes 8x8 alternating grid
  return kernel

def getTile(img, i,j,tile_res):
  """Assumes no buffer in image"""
  return img[tile_res*i:tile_res*(i+1),tile_res*j:tile_res*(j+1)]


if __name__ == '__main__':
  PLOT_RESULTS = True

  input_folder = "rectified"

  tile_res = 64
  tile_buffer = 1

  side_len = 8*tile_res
  buffer_size = tile_buffer*tile_res

  filename ="%d.png" % 31
  filepath = "%s/%s" % (input_folder,filename)
  print("Segmenting %s..." % filename)
  img_orig = np.array(PIL.Image.open(filepath).convert('RGB'))

  img_h, img_w, _ = img_orig.shape

  # Bilateral smooth image
  img = img_orig
  bg_illum = cv2.blur(img,ksize=(tile_res*4,tile_res*4))
  img = (bg_illum.mean() + (img.astype(np.float64) - bg_illum)).astype(np.uint8)
  # img = cv2.blur(img,ksize=(5,5)) 
  # img = cv2.medianBlur(img,7)
  # img = cv2.bilateralFilter(img,int(tile_res/4),75,75) 
  
  ideal_corners = getIdealCorners(tile_res, tile_buffer)

  img_checkerboard = img[buffer_size:-buffer_size, buffer_size:-buffer_size]
  img_checkerboard_before = img_checkerboard.copy()

  ycrcb = cv2.cvtColor(img_checkerboard, cv2.COLOR_RGB2YCR_CB)
  # ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0].astype(np.uint8))
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
  ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0].astype(np.uint8))
  img_checkerboard = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)
  img_checkerboard = cv2.medianBlur(img_checkerboard,7)

  img_checkerboard_gray = np.array(PIL.Image.fromarray(img_checkerboard).convert('L'))
  # img_checkerboard_gray = cv2.equalizeHist(img_checkerboard_gray)

  img_draw = np.zeros(img_checkerboard_gray.shape)

  # Watershed
  markers = np.zeros(img_checkerboard_gray.shape, dtype=np.int32)
  teb = int(64/2-2)
  N = 8
  marker_counter = 2
  # watershed_mask = np.zeros(markers.shape, dtype=bool)
  for i in range(N):
    # watershed_mask[tile_res*i,:] = False
    # watershed_mask[:,tile_res*i] = False
    for j in range(N):
      markers[tile_res*i+teb:tile_res*(i+1)-teb,tile_res*j+teb:tile_res*(j+1)-teb] = marker_counter
      marker_counter += 1
  
  markers_init = markers.copy()
  img_watershed = cv2.watershed(img_checkerboard, markers)
  # from skimage.morphology import watershed
  # img_watershed = watershed(img_checkerboard_gray, markers, mask=~watershed_mask)


  if PLOT_RESULTS:
    print("Plotting")
    plt.figure(filename, figsize=(20,8))

    teb = 10
    N = 8
    for i in range(N):
      for j in range(N):
        k = i*N + j + 1
        # plt.subplot(N,N,k)

        tile = getTile(img_checkerboard,i,j,tile_res)
        tile_gray = getTile(img_checkerboard_gray,i,j,tile_res)[teb:-teb,teb:-teb]
        _,tile_thresh = cv2.threshold(tile_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        tile_draw = np.ma.masked_where(tile_thresh, tile_gray)
        if np.std(tile_gray) > 10:
          img_draw[tile_res*i+teb:tile_res*(i+1)-teb,tile_res*j+teb:tile_res*(j+1)-teb] = tile_thresh
        #   plt.imshow(tile_draw)
        # plt.axis([0,tile_res,tile_res,0])
        # plt.axis('off')
    plt.subplot(241)
    plt.imshow(img_orig)
    plt.subplot(242)
    plt.imshow(img_checkerboard_before)
    plt.title('Checkerboard Before')
    plt.subplot(243)
    plt.imshow(img_draw)
    plt.title('Thresholded')
    plt.subplot(244)
    plt.imshow(img_watershed)
    plt.title('Watershed')
    plt.subplot(245)
    plt.imshow(img_checkerboard_gray, cmap='Greys_r')
    plt.title('Gray')
    plt.subplot(246)
    plt.imshow(img_checkerboard)
    plt.title('Checkerboard After')

    plt.subplot(247)
    plt.imshow(markers_init)
    plt.title('Markers init')

    plt.subplot(248)
    plt.imshow(bg_illum)
    plt.title('Background Illumination')


    plt.show()