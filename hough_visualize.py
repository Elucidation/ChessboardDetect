from __future__ import print_function
from pylab import *
import numpy as np
import cv2
import sys
from board_detect import *
from contour_detect import *
from rectify_refine import *


def getRhoTheta(line):
  x1,y1,x2,y2 = line
  theta = np.arctan2(y2-y1,x2-x1)
  rho = x1*np.cos(theta) + y1*sin(theta)
  return rho, theta


def findAndDrawTile(img):
  contours, chosen_tile_idx, edges = findPotentialTiles(img)
  if not len(contours):
    return
  drawPotentialTiles(img, contours, chosen_tile_idx)

  tile_corners = getChosenTile(contours, chosen_tile_idx)

  hough_corners, corner_hough_lines, edges_roi = refineTile(img, edges, contours, chosen_tile_idx)
  drawBestHoughLines(img, hough_corners, corner_hough_lines)

  # Single tile warp
  ideal_tile = np.array([
      [1,0],
      [1,1],
      [0,1],
      [0,0],
      ],dtype=np.float32)
  tile_res=32
  M = cv2.getPerspectiveTransform(hough_corners,
                                  (tile_res)*(ideal_tile+8+1))
  side_len = tile_res*(8 + 1)*2
  out_img = cv2.warpPerspective(img, M,
                                (side_len, side_len))

  cv2.imshow('image %dx%d' % (img.shape[1],img.shape[0]),img)
  cv2.imshow('warp',out_img)


def findAndDrawHough(img):
  img_diag_size = int(np.ceil(np.sqrt(img.shape[0]*img.shape[0] + img.shape[1]*img.shape[1])))

  hough_img = np.zeros([2*img_diag_size/4, 180]) # -90 to 90 deg, -rho_max to rho_max

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray,100,650,apertureSize = 3)
  min_img_side = min(img.shape[:2])
  minLineLength = min_img_side/4
  maxLineGap = min_img_side/10
  threshold = int(min_img_side/4)
  # print(minLineLength, maxLineGap, threshold)
  lines = cv2.HoughLinesP(edges,rho=1,theta=np.pi/180,
    threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

  if any(lines):
    rhothetas = np.zeros([lines.shape[0], 2])
    for i, (x1,y1,x2,y2) in enumerate(lines[:,0,:]):
      cv2.line(img,(x1,y1),(x2,y2), (0,0,255),2)

      rho, theta = getRhoTheta((x1,y1,x2,y2))
      rhothetas[i,:] = rho, theta
      img_rho, img_theta = (int(theta*180/np.pi + 90), int((rho+img_diag_size)/4))
      cv2.circle(hough_img, (img_rho, img_theta), 3, (255,0,0),-1)



  
  # Using opencv
  cv2.imshow('image %dx%d' % (img.shape[1],img.shape[0]),img)
  cv2.imshow('edges',edges)
  cv2.imshow('hough',hough_img)

def findAndDrawMask(img):
  # img = scaleImageIfNeeded(img, 600, 480)

  # Edges
  edges = cv2.Canny(img, 100, 550)
  mask, top_two_angles, min_area_rect, median_contour = getEstimatedChessboardMask(img, edges, iters=5)

  img_masked_full = cv2.bitwise_and(img,img,mask = (mask > 0.5).astype(np.uint8))
  img_masked = cv2.addWeighted(img,0.2,img_masked_full,0.8,0)

  # Hough lines overlay
  edges_masked = cv2.bitwise_and(edges,edges,mask = (mask > 0.5).astype(np.uint8))
  
  if top_two_angles is not None and len(top_two_angles) == 2:
    lines = getHoughLines(edges_masked, min_line_size=0.25*min(min_area_rect[1]))
    lines_a, lines_b = parseHoughLines(lines, top_two_angles, angle_threshold_deg=15)
    
    plotHoughLines(img_masked, lines, color=(255,255,255), line_thickness=1)
    plotHoughLines(img_masked, lines_a, color=(0,0,255))
    plotHoughLines(img_masked, lines_b, color=(0,255,0))

  if min_area_rect is not None:
    drawMinAreaRect(img_masked, min_area_rect)

  # cv2.imshow('Masked',img_masked)
  return img_masked
  # cv2.imshow('edges %s' % filename, edges_masked)
  # cv2.imshow('mask %s' % filename, mask)

def findAndDrawChessboard(img):
  img_orig = img.copy()
  img_orig2 = img.copy()

  # Edges
  edges = cv2.Canny(img, 100, 550)

  # Get mask for where we think chessboard is
  mask, top_two_angles, min_area_rect, median_contour = getEstimatedChessboardMask(img, edges,iters=3) # More iters gives a finer mask
  if top_two_angles is None or len(top_two_angles) != 2 or min_area_rect is None:
    print('fail', top_two_angles)
    return img

  if mask.min() != 0:
    return img

  # Get hough lines of masked edges
  edges_masked = cv2.bitwise_and(edges,edges,mask = (mask > 0.5).astype(np.uint8))
  img_orig = cv2.bitwise_and(img_orig,img_orig,mask = (mask > 0.5).astype(np.uint8))

  lines = getHoughLines(edges_masked, min_line_size=0.25*min(min_area_rect[1]))

  lines_a, lines_b = parseHoughLines(lines, top_two_angles, angle_threshold_deg=35)
  if len(lines_a) < 2 or len(lines_b) < 2:
    print('fail2', lines_a, lines_b)
    return img
  
  # plotHoughLines(img, lines, color=(255,255,255), line_thickness=1)
  # plotHoughLines(img, lines_a, color=(0,0,255))
  # plotHoughLines(img, lines_b, color=(0,255,0))

  a = time()
  for i2 in range(2):
    for i in range(5):
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
        break
      # else:
      #   print('iter %d: %.2f %.2f' % (i, abs(median_theta_a), abs(median_theta_b)))

    warp_img, M = getTileImage(img_orig, corners.astype(np.float32),tile_buffer=16, tile_res=16)

    lines_x, lines_y, step_x, step_y = getWarpCheckerLines(warp_img)
    if len(lines_x) > 0:
      break

  warp_img, M = getTileImage(img_orig, corners.astype(np.float32),tile_buffer=16, tile_res=16)

  for corner in corners:
      cv2.circle(img, tuple(map(int,corner)), 5, (255,150,150),-1)  

  if len(lines_x) > 0:
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

    cv2.polylines(img, [real_corners.astype(np.int32)], True, (150,50,255), thickness=4)
    cv2.polylines(img, [all_real_tile_centers.astype(np.int32)], False, (0,50,255), thickness=1)
    

  img_masked_full = cv2.bitwise_and(img,img,mask = (mask > 0.5).astype(np.uint8))
  img_masked = cv2.addWeighted(img,0.2,img_masked_full,0.8,0)

  drawMinAreaRect(img_masked, min_area_rect)

  return img_masked


def processVideo(filename, func=findAndDrawHough, rate=1):

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  cap = cv2.VideoCapture(filename)

  img_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  img_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  img_rescale_ratio = 1.0

  out_size = (int(img_width*img_rescale_ratio), int(img_height*img_rescale_ratio))

  output_filename = 'output3_%s.avi' % (filename[:-4])
  print("Writing to %s at scale %s" % (output_filename, out_size))
  # out = cv2.VideoWriter(output_filename,fourcc, 20.0, (384,216)) # 0.2
  # out = cv2.VideoWriter(output_filename,fourcc, 20.0, (576,324)) # 0.3
  out = cv2.VideoWriter(output_filename,fourcc, 20.0, out_size)

  i = 0
  while(cap.isOpened()):
    ret, frame = cap.read()
    if not np.any(frame):
      break
    i+=1
    

    img = cv2.resize(frame,None,fx=img_rescale_ratio, fy=img_rescale_ratio, interpolation = cv2.INTER_AREA)
    
    if (i == 1):
      if img.shape[0] != out_size[1] or img.shape[1] != out_size[0]:
        print(img.shape, out_size)
    
    img_masked = func(img)

    out.write(img_masked)
    cv2.imshow('Masked',img_masked)

    if cv2.waitKey(rate) & 0xFF == ord('q'):
      break
  
  cap.release()
  out.release()
  cv2.destroyAllWindows()

def main(filenames):
  for filename in filenames:
    print("Processing %s" % filename)
    img = cv2.imread(filename)

    img_diag_size = int(np.ceil(np.sqrt(img.shape[0]*img.shape[0] + img.shape[1]*img.shape[1])))
    print(img_diag_size)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,650,apertureSize = 3)
    min_img_side = min(img.shape[:2])
    minLineLength = min_img_side/8
    maxLineGap = min_img_side/10
    threshold = int(min_img_side/8)
    print(minLineLength, maxLineGap, threshold)
    lines = cv2.HoughLinesP(edges,rho=1,theta=np.pi/180,
      threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    # colors = np.random.random([lines.shape[0],3])*255
    colors = [
              [255,0,0],
              [0,255,0],
              [255,255,0],
              [0,0,255],
              [255,0,255],
              [0,255,255],
              [255,255,255],
             ]


    hough_img = np.zeros([2*img_diag_size/4, 180]) # -90 to 90 deg, -rho_max to rho_max

    if any(lines):
      rhothetas = np.zeros([lines.shape[0], 2])
      for i, (x1,y1,x2,y2) in enumerate(lines[:,0,:]):
        color = list(map(int,colors[i%len(colors)])) # dtype needs to be int, not np.int32
        cv2.line(img,(x1,y1),(x2,y2), color,2)

        rho, theta = getRhoTheta((x1,y1,x2,y2))
        rhothetas[i,:] = rho, theta
        img_rho, img_theta = (int(theta*180/np.pi + 90), int((rho+img_diag_size)/4))
        cv2.circle(hough_img, (img_rho, img_theta), 3, (255,0,0),-1)

      plot(rhothetas[:,1]*180/np.pi, rhothetas[:,0], 'o')
      print(hough_img.shape)
      xlabel('theta (deg)')
      ylabel('rho')



    
    # Using matplotlib
    # imshow(img)
    # show()
    
    # Using opencv
    cv2.imshow('image %dx%d' % (img.shape[1],img.shape[0]),img)
    cv2.imshow('edges',edges)
    cv2.imshow('hough',hough_img)
    cv2.moveWindow('hough', 0,0)
    # cv2.waitKey(0)
    axis('equal')
    show()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  if len(sys.argv) > 1:
    filenames = sys.argv[1:]
  else:
    filenames = ['input2/27.jpg']
    # filenames = ['input/2.jpg', 'input/6.jpg', 'input/17.jpg']
    # filenames = ['input/1.jpg', 'input/2.jpg', 'input/3.jpg', 'input_fails/37.jpg', 'input_fails/38.jpg']
    # filenames = ['input_fails/37.jpg', 'input_fails/38.jpg']
  # main(filenames)
  # processVideo('chess1.mp4')
  # processVideo('chess2.mp4', func=findAndDrawTile)
  processVideo('chess1.mp4', func=findAndDrawMask)
  # processVideo('chess3.mp4', func=findAndDrawChessboard)
  print('Done.')