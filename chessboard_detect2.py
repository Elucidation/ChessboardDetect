from __future__ import print_function
import cv2
import PIL.Image
import numpy as np
import sys
from time import time
from matplotlib import pyplot as plt
from contour_detect import  *
from line_intersection import  *
from rectify_refine import *

np.set_printoptions(suppress=True, precision=2, linewidth=200)

def processFile(filename):
  img = cv2.imread(filename)
  # img = scaleImageIfNeeded(img, 600, 480)
  img = scaleImageIfNeeded(img, 1024, 768)
  img_orig = img.copy()
  img_orig2 = img.copy()

  # Edges
  edges = cv2.Canny(img, 100, 550)

  # Get mask for where we think chessboard is
  mask, top_two_angles, min_area_rect, median_contour = getEstimatedChessboardMask(img, edges,iters=3) # More iters gives a finer mask
  print("Top two angles (in image coord system): %s" % top_two_angles)

  # Get hough lines of masked edges
  edges_masked = cv2.bitwise_and(edges,edges,mask = (mask > 0.5).astype(np.uint8))
  img_orig = cv2.bitwise_and(img_orig,img_orig,mask = (mask > 0.5).astype(np.uint8))

  lines = getHoughLines(edges_masked, min_line_size=0.25*min(min_area_rect[1]))
  print("Found %d lines." % len(lines))

  lines_a, lines_b = parseHoughLines(lines, top_two_angles, angle_threshold_deg=35)
  
  # plotHoughLines(img, lines, color=(255,255,255), line_thickness=1)
  # plotHoughLines(img, lines_a, color=(0,0,255))
  # plotHoughLines(img, lines_b, color=(0,255,0))
  if len(lines_a) < 2 or len(lines_b) < 2:
    return img_orig, edges_masked, img_orig

  a = time()
  for i2 in range(10):
    for i in range(100):
      corners = chooseRandomGoodQuad(lines_a, lines_b, median_contour)
      
      # warp_img, M = getTileImage(img_orig, corners.astype(np.float32),tile_buffer=16, tile_res=16)
      M = getTileTransform(corners.astype(np.float32),tile_buffer=16, tile_res=16)

      # Warp lines and draw them on warped image
      all_lines = np.vstack([lines_a[:,:2], lines_a[:,2:], lines_b[:,:2], lines_b[:,2:]]).astype(np.float32)
      warp_pts = cv2.perspectiveTransform(all_lines[None,:,:], M)
      warp_pts = warp_pts[0,:,:]
      warp_lines_a = np.hstack([warp_pts[:len(lines_a),:], warp_pts[len(lines_a):2*len(lines_a),:]])
      warp_lines_b = np.hstack([warp_pts[2*len(lines_a):2*len(lines_a)+len(lines_b),:], warp_pts[2*len(lines_a)+len(lines_b):,:]])


      # Get thetas of warped lines 
      thetas_a = np.array([getSegmentTheta(line) for line in warp_lines_a])
      thetas_b = np.array([getSegmentTheta(line) for line in warp_lines_b])
      median_theta_a = (np.median(thetas_a*180/np.pi))
      median_theta_b = (np.median(thetas_b*180/np.pi))
      
      # Gradually relax angle threshold over N iterations
      if i < 20:
        warp_angle_threshold = 0.03
      elif i < 30:
        warp_angle_threshold = 0.1
      elif i < 50:
        warp_angle_threshold = 0.3
      elif i < 70:
        warp_angle_threshold = 0.5
      elif i < 80:
        warp_angle_threshold = 1.0
      else:
        warp_angle_threshold = 2.0
      if ((angleCloseDeg(abs(median_theta_a), 0, warp_angle_threshold) and 
            angleCloseDeg(abs(median_theta_b), 90, warp_angle_threshold)) or 
          (angleCloseDeg(abs(median_theta_a), 90, warp_angle_threshold) and 
            angleCloseDeg(abs(median_theta_b), 0, warp_angle_threshold))):
        print('Found good match (%d): %.2f %.2f' % (i, abs(median_theta_a), abs(median_theta_b)))
        break
      # else:
      #   print('iter %d: %.2f %.2f' % (i, abs(median_theta_a), abs(median_theta_b)))

    warp_img, M = getTileImage(img_orig, corners.astype(np.float32),tile_buffer=16, tile_res=16)

    # Recalculate warp now that we're using a different tile_buffer/res
    # warp_pts = cv2.perspectiveTransform(all_lines[None,:,:], M)
    # warp_pts = warp_pts[0,:,:]
    # warp_lines_a = np.hstack([warp_pts[:len(lines_a),:], warp_pts[len(lines_a):2*len(lines_a),:]])
    # warp_lines_b = np.hstack([warp_pts[2*len(lines_a):2*len(lines_a)+len(lines_b),:], warp_pts[2*len(lines_a)+len(lines_b):,:]])
    
    lines_x, lines_y, step_x, step_y = getWarpCheckerLines(warp_img)
    if len(lines_x) > 0:
      print('Found good chess lines (%d): %s %s' % (i2, lines_x, lines_y))
      break
  print("Ransac corner detection took %.4f seconds." % (time() - a))

  print(lines_x, lines_y)
  warp_img, M = getTileImage(img_orig, corners.astype(np.float32),tile_buffer=16, tile_res=16)

  for corner in corners:
      cv2.circle(img, tuple(map(int,corner)), 5, (255,150,150),-1)  

  if len(lines_x) > 0:
    print('Found chessboard?')
    warp_corners, all_warp_corners = getRectChessCorners(lines_x, lines_y)
    tile_centers = all_warp_corners + np.array([step_x/2.0, step_y/2.0]) # Offset from corner to tile centers
    M_inv = np.matrix(np.linalg.inv(M))
    real_corners, all_real_tile_centers = getOrigChessCorners(warp_corners, tile_centers, M_inv)

    tile_res = 64 # Each tile has N pixels per side
    tile_buffer = 1
    warp_img, better_M = getTileImage(img_orig2, real_corners, tile_buffer=tile_buffer, tile_res=tile_res)
    # Further refine rectified image
    warp_img, was_rotated, refine_M = reRectifyImages(warp_img)
    # combined_M = better_M
    combined_M = np.matmul(refine_M,better_M)
    M_inv = np.matrix(np.linalg.inv(combined_M))

    # Get better_M based corners
    hlines = vlines = (np.arange(8)+tile_buffer)*tile_res
    hcorner = (np.array([0,8,8,0])+tile_buffer)*tile_res
    vcorner = (np.array([0,0,8,8])+tile_buffer)*tile_res
    ideal_corners = np.vstack([hcorner,vcorner]).T
    ideal_all_corners = np.array(list(itertools.product(hlines, vlines)))
    ideal_tile_centers = ideal_all_corners + np.array([tile_res/2.0, tile_res/2.0]) # Offset from corner to tile centers

    real_corners, all_real_tile_centers = getOrigChessCorners(ideal_corners, ideal_tile_centers, M_inv)
    
    # Get final refined rectified warped image for saving
    warp_img, _ = getTileImage(img_orig2, real_corners, tile_buffer=tile_buffer, tile_res=tile_res)

    cv2.polylines(img, [real_corners.astype(np.int32)], True, (150,50,255), thickness=3)
    cv2.polylines(img, [all_real_tile_centers.astype(np.int32)], False, (0,50,255), thickness=1)
    
    # Update mask with predicted chessboard
    cv2.drawContours(mask,[real_corners.astype(int)],0,1,-1)


  img_masked_full = cv2.bitwise_and(img,img,mask = (mask > 0.5).astype(np.uint8))
  img_masked = cv2.addWeighted(img,0.2,img_masked_full,0.8,0)

  drawMinAreaRect(img_masked, min_area_rect)

  return img_masked, edges_masked, warp_img


def other():
  # vals = np.array([224, 231, 238, 257, 271, 278, 300, 321, 342, 358, 362, 383, 404, 425, 436, 463, 474])
  # vals_wrong = np.array([ 257., 278., 300., 321., 342., 358., 362., 383., 404.])
  # vals = np.array([206, 222, 239, 256, 268, 273, 286, 290, 307, 324, 341, 345, 357, 373])
  # vals_wrong = np.array([ 226.5, 239., 256., 268., 273., 286., 290., 307., 319.5])
  # vals = np.array([252, 260, 272, 278, 294, 300, 314, 336, 357, 379, 400])
  # vals = np.array([272, 283, 298, 306, 324, 331, 349, 374, 399, 424, 449])
  # vals = np.array([13, 29, 49, 64, 82, 88, 96, 150, 159, 167, 179, 204, 212, 218, 228, 235, 247, 260, 272, 285, 305, 338, 363, 370, 380, 389, 402, 411, 432, 463, 478])
  vals = np.array([67, 93, 100, 111, 122, 140, 147, 158, 172, 184, 209, 219, 228, 237, 249, 273, 298, 317, 324, 344, 349, 356, 374, 400, 414, 426])

  print(vals)
  print(np.diff(vals))
  # sub_arr = np.abs(vals[:,None] - vals)
  # print(sub_arr)

  n_pts = 3
  n = scipy.special.binom(len(vals),n_pts)
  # devs = np.zeros(n)
  # plt.plot(vals_wrong,np.zeros(len(vals_wrong)),'rs')

  a = time()
  best_spacing = getBestEqualSpacing(vals)
  print("iter cost took %.4f seconds for %d combinations." % (time() - a, n))
  print(best_spacing)
  plt.plot(best_spacing,0.05+np.zeros(len(best_spacing)),'gx')
  
  # plt.hist(devs, 50)

  plt.plot(vals,-0.1 + np.zeros(len(vals)),'k.', ms=10)
  plt.show()



def main(filenames):
  for filename in filenames:
    a = time()
    img_masked, edges_masked, warp_img = processFile(filename)
    print("Full image file process took %.4f seconds." % (time() - a))
    cv2.imshow('img %s' % filename,img_masked)
    cv2.imshow('warp %s' % filename, warp_img)
    out_filename = filename[:-4].replace('/','_').replace('\\','_')
    print(filename[:-4], out_filename)
    PIL.Image.fromarray(cv2.cvtColor(warp_img,cv2.COLOR_BGR2RGB)).save("rectified2/%s.png" % out_filename)
    # cv2.imshow('edges %s' % filename, edges_masked)


  cv2.waitKey(0)
  cv2.destroyAllWindows()
  plt.show()

if __name__ == '__main__':
  if len(sys.argv) > 1:
    filenames = sys.argv[1:]
  else:
    # filenames = ['input2/02.jpg']
    # filenames = ['input2/01.jpg']
    filenames = ['input/30.jpg']
  print("Loading", filenames)
  main(filenames)
  # other()