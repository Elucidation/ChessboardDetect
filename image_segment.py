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

  filename ="%d.png" % 7
  filepath = "%s/%s" % (input_folder,filename)
  print("Segmenting %s..." % filename)
  img_orig = np.array(PIL.Image.open(filepath).convert('RGB'))

  img_h, img_w, _ = img_orig.shape

  # Bilateral smooth image
  img = img_orig
  # img = cv2.blur(img,ksize=(5,5)) 
  # img = cv2.bilateralFilter(img,int(tile_res/4),75,75) 
  
  ideal_corners = getIdealCorners(tile_res, tile_buffer)
  ideal_checkerboard = getIdealCheckerboardPattern(tile_res, tile_buffer)
  ideal_checkerboard_corners = getIdealCorners(tile_res, 0)

  white_only_mask = ideal_checkerboard
  black_only_mask = (~white_only_mask.astype(bool)).astype(np.uint8)

  img_checkerboard = img[buffer_size:-buffer_size, buffer_size:-buffer_size]
  img_checkerboard = cv2.medianBlur(img_checkerboard,7)
  img_checkerboard = cv2.bilateralFilter(img_checkerboard,int(tile_res/4),75,75) 

  # Local Histogram Equalization of checkerboard
  ycrcb = cv2.cvtColor(img_checkerboard, cv2.COLOR_RGB2YCR_CB)
  # ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0].astype(np.uint8))
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
  # ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0].astype(np.uint8))
  img_checkerboard = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)
  # ycrcb = cv2.medianBlur(ycrcb,7)
  # ycrcb = cv2.bilateralFilter(ycrcb,int(tile_res/4),75,75) 


  responseA = cv2.bitwise_and(img_checkerboard, img_checkerboard, mask=white_only_mask)
  responseB = cv2.bitwise_and(img_checkerboard, img_checkerboard, mask=black_only_mask)

  img_checkerboard_gray = np.array(PIL.Image.fromarray(img_checkerboard).convert('L'))
  # img_checkerboard_gray = cv2.equalizeHist(img_checkerboard_gray)
  # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
  # img_checkerboard_gray = clahe.apply(img_checkerboard_gray)

  # Get La*b* colorspace
  # lab = color.rgb2lab(img_checkerboard)
  # hsv = color.rgb2hsv(img_checkerboard)

  # Remove intensity changes
  # Local Histogram Equalization
  # lab[:,:,0] = cv2.equalizeHist(lab[:,:,0].astype(np.uint8))
  # lab[:,:,0] = exposure.equalize_adapthist(lab[:,:,0]*0.01, clip_limit=0.3)*100
  # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
  # lab[:,:,0] = clahe.apply(lab[:,:,0].astype(np.uint8))

  # img_checkerboard_equalized = color.lab2rgb(lab)

  # edges = cv2.Canny(lab[:,:,0].astype(np.uint8),200,500,apertureSize = 3, L2gradient=False) # Better thresholds
  low_threshold = 30
  edges = cv2.Canny(img_checkerboard_gray,low_threshold,3*low_threshold,apertureSize = 3, L2gradient=False) # Better thresholds
  

  rgb_flat = img_checkerboard.reshape([img_checkerboard.shape[0]*img_checkerboard.shape[1],img_checkerboard.shape[2]])
  # rgb_flat[white_only_mask.astype(bool).flatten()] = 0
  # lab_flat = lab.reshape([lab.shape[0]*lab.shape[1],lab.shape[2]])
  # hsv_flat = hsv.reshape([hsv.shape[0]*hsv.shape[1],hsv.shape[2]])

  # a_star_white = lab[:,:,1][white_only_mask.astype(bool)]
  # b_star_white = lab[:,:,2][white_only_mask.astype(bool)]

  # a_star_black = lab[:,:,1][black_only_mask.astype(bool)]
  # b_star_black = lab[:,:,2][black_only_mask.astype(bool)]

  # a_star = lab[:,:,1].flatten()
  # b_star = lab[:,:,2].flatten()

  # K-means cluster into 4 parts (black tile, white tile, black piece, white piece)

  print("Start K-means")
  random_state = 1
  clt = KMeans(n_clusters=4, random_state=random_state)
  clt.fit(ycrcb[:,:,1].reshape([ycrcb.shape[0]*ycrcb.shape[1],-1]))
  # clt.fit(ycrcb.reshape([ycrcb.shape[0]*ycrcb.shape[1],-1]))
  # clt.fit(img_checkerboard.reshape([img_checkerboard.shape[0]*img_checkerboard.shape[1],-1]))
  y_pred = clt.labels_
  print("End K-means", y_pred.shape)

  deviations = np.zeros([8,8])
  for i in range(8):
    for j in range(8):
      tile = getTile(img_checkerboard_gray,i,j,tile_res)
      edge_tile = getTile(edges,i,j,tile_res)
      inner_tile = tile[8:-8,8:-8]
      inner_edge_tile = edge_tile[16:-16,16:-16]
      if np.sum(inner_edge_tile) > 20:
        deviations[i,j] = np.std(inner_tile)



  if PLOT_RESULTS:
    print("Plotting")
    plt.figure(filename)

    plt.subplot(331)
    plt.imshow(img_orig)
    plt.plot(ideal_corners[:,0], ideal_corners[:,1], 'ro', ms=3)
    plt.title('Input rectified image')
    plt.axis([0,img_w,img_h, 0])    

    plt.subplot(332)
    plt.imshow(responseA)
    plt.plot(ideal_checkerboard_corners[:,0], ideal_checkerboard_corners[:,1], 'ro', ms=3)
    plt.title('White chessboard only')
    plt.axis([0,side_len, side_len, 0])
    
    plt.subplot(333)
    plt.imshow(img_checkerboard_gray,cmap=plt.cm.gray)
    # plt.imshow(responseB)
    # plt.plot(ideal_checkerboard_corners[:,0], ideal_checkerboard_corners[:,1], 'ro', ms=3)
    # plt.title('Black chessboard only')
    plt.axis([0,side_len, side_len, 0])

    plt.subplot(334)
    plt.imshow(img_checkerboard)
    plt.axis([0,side_len, side_len, 0])
    
    # plt.hist2d(a_star, b_star, (100,100))
    # plt.title('La*b* : all')
    # plt.xlabel('A*')
    # plt.ylabel('B*')
    # plt.colorbar()

    plt.subplot(335)
    plt.imshow(edges)
    plt.axis([0,side_len, side_len, 0])

    # Plot only random N points
    # subset = np.random.choice(a_star.shape[0], 2000,replace=False)
    # x = np.hstack([np.ones([subset.shape[0],1])*50, clt.cluster_centers_[y_pred[subset]]])
    # x = np.swapaxes(np.atleast_3d(x),1,2)
    # x = np.squeeze(color.lab2rgb(x))
    # plt.scatter(a_star[subset], b_star[subset], c=y_pred[subset], edgecolor='')

    plt.subplot(336)
    plt.imshow(img_checkerboard)
    plt.plot(ideal_checkerboard_corners[:,0], ideal_checkerboard_corners[:,1], 'ro', ms=3)
    for i in range(8):
      for j in range(8):
        if deviations[i,j] > 8:
          plt.text(tile_res*(j+0.5)-20, tile_res*(i+0.5),
            '%.1f' % deviations[i,j], color='black', size=10, fontweight='heavy');
        

    plt.title('Standard deviation of tiles')
    plt.axis('square')
    plt.axis([0,side_len, side_len, 0])

    plt.subplot(337)
    kmeans_result = y_pred.reshape(ideal_checkerboard.shape)
    plt.imshow(kmeans_result)
    plt.colorbar()

    plt.subplot(338)
    kmeans_result_fix = kmeans_result.copy()
    kmeans_result_fix[white_only_mask.astype(bool)] = (~kmeans_result_fix[white_only_mask.astype(bool)].astype(bool)).astype(np.uint8)
    plt.imshow(kmeans_result_fix)
    plt.colorbar()

    plt.subplot(339)
    plt.imshow(ycrcb[:,:,1])
    plt.colorbar()

    plt.show()