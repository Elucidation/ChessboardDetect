import PIL.Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import itertools
import os
np.set_printoptions(precision=2, linewidth=400, suppress=True) # Better printing of arrays

def nonMaxSupress2D(img, win=10):
  out_img = img.copy()
  w, h = img.shape
  for i in range(w):
    for j in range(h):
      # Skip if already suppressed
      # if out_img[i,j] == 0:
      #   continue
      # Get neigborhood
      ta=max(0,i-win)
      tb=min(w,i+win+1)
      tc=max(0,j-win)
      td=min(h,j+win+1)
      cell = img[ta:tb,tc:td]
      # Blank out all non-max values in window
      out_img[ta:tb,tc:td] = (cell == cell.max()) * out_img[ta:tb,tc:td]
  return out_img

def reRectifyImages(color_img, tile_res=64, tile_buffer=1):
  """Recenter off-by-N-tiles chessboards, then use 
  corner subpixel with ransac findHomography to further optimize rectification of image"""
  if type(color_img).__module__ == np.__name__:
    img = np.array(PIL.Image.fromarray(color_img).convert('L')) # grayscale uint8 numpy array
  else:
    img = np.array(color_img.convert('L')) # grayscale uint8 numpy array
  # img = cv2.equalizeHist(img) # Global histogram equalization
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  img = clahe.apply(img)
  # Make sure our tile resolutions and buffer expectations match image size
  assert( (img.shape[0] / tile_res) == (8+2*tile_buffer) )

  k = tile_res
  side_len = tile_res*(8+2*tile_buffer)
  quad = np.ones([k,k])
  # kernel = np.vstack([np.hstack([quad,-quad]), np.hstack([-quad,quad])])
  kernel = np.vstack([np.hstack([quad,-quad]), np.hstack([-quad,quad])])
  kernel = np.tile(kernel,(4,4)) # Becomes 8x8 alternating grid
  kernel = kernel/np.linalg.norm(kernel)

  response = cv2.filter2D(img, cv2.CV_32F, kernel)
  corners = abs(response)
  expected_ctr_pt = np.array([tile_res*(4+tile_buffer), tile_res*(4+tile_buffer)])
  best_pt = np.array(np.unravel_index(corners.argmax(), corners.shape))[::-1] # Flipped due to image y/x
  center_offset = best_pt - expected_ctr_pt

  # print(best_pt)
  should_rotate = response[best_pt[1],best_pt[0]] < 0
  # print(should_rotate)

  hlines = best_pt[0] + (np.arange(9)-4)*tile_res
  vlines = best_pt[1] + (np.arange(9)-4)*tile_res

  all_corners = np.array(list(itertools.product(hlines, vlines)))
  criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
  better_corners = cv2.cornerSubPix(img, all_corners.astype(np.float32),
    (20,20), (-1,-1), criteria)

  M, good_pts = cv2.findHomography(better_corners.astype(np.float32) + center_offset, all_corners.astype(np.float32), cv2.RANSAC)

  # should_rotate = checkChessboardAlignment(img)
  
  color_img_arr = np.array(color_img)

  if (should_rotate):
    # color_img_arr = np.rot90(color_img_arr)
    # rot90 = cv2.getRotationMatrix2D((side_len/2,side_len/2),90,1)
    rot90 = np.matrix([[0, 1, 0], [-1, 0, side_len], [0, 0, 1]])
    M = np.matmul(rot90, M)

  return cv2.warpPerspective(color_img_arr, M, color_img_arr.shape[:2]), should_rotate, M

# def checkChessboardAlignment(img,  tile_res=64, tile_buffer=1):
  # Guess if chessboard black/white corners are correct, rotate 90 deg otherwise 

if __name__ == '__main__':
  PLOT_RESULTS = True

  input_folder = "rectified"

  tile_res = 64
  tile_buffer = 1

  filename ="%d.png" % 7
  filepath = "%s/%s" % (input_folder,filename)
  print("Refining %s" % filename)
  img_orig = PIL.Image.open(filepath)

  # Grayscale
  better_img, was_rotated, refine_M = reRectifyImages(img_orig)
  # print(refine_M)
  if was_rotated:
    print("Was rotated")

  if PLOT_RESULTS:
    hlines = vlines = (np.arange(9)+tile_buffer)*tile_res
    ideal_corners = np.array(list(itertools.product(hlines, vlines)))

    plt.subplot(121)
    plt.imshow(img_orig)
    plt.plot(ideal_corners[:,0], ideal_corners[:,1], 'ro', ms=3)
    plt.title('Input rectified image')
    
    plt.subplot(122)
    plt.imshow(better_img)
    plt.plot(ideal_corners[:,0], ideal_corners[:,1], 'ro', ms=3)
    plt.title('Refined output rectified image')

    plt.show()