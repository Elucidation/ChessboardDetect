import cv2 # For Sobel etc
import numpy as np
from helpers import *
from line_intersection import *

def getChessLinesCorners(img):
  # Edges
  edges = cv2.Canny(img,200,500,apertureSize = 3, L2gradient=False) # Better thresholds

  # Gradients
  sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
  sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
  grad_mag = np.sqrt(sobelx**2+sobely**2)

  # Hough Lines Probabilistic
  chessboard_to_screen_ratio = 0.2
  min_chessboard_line_length = chessboard_to_screen_ratio * min(img.shape)
  # TODO: This varys based on the chessboard to screen ratio, for chessboards filling the screen, we want to hop further
  max_line_gap = min_chessboard_line_length / 8.0 * 1.5 # Can hop up to one missing square

  lines = cv2.HoughLinesP(edges,1,np.pi/360.0, 30, minLineLength = min_chessboard_line_length, maxLineGap = max_line_gap)[:,0,:]
  # lines = cv2.HoughLinesP(edges,1,np.pi/360.0, 30, minLineLength=30, maxLineGap=30)[:,0,:]

  good_lines = np.zeros(lines.shape[0])
  norm_grads = np.zeros(lines.shape[0])
  angles = np.zeros(lines.shape[0])

  for idx in range(lines.shape[0]):
    line = lines[idx,:]
    is_good, _, _, _, _, avg_normal_gradient = getLineGradients(line, sobelx, sobely, grad_mag)
    good_lines[idx] = is_good
    norm_grads[idx] = avg_normal_gradient
    angles[idx] = getSegmentAngle(line)

  # Get angles and segment lines up
  segments = segmentAngles(angles, good_lines, angle_threshold=15*np.pi/180)
  top_two_segments = chooseBestSegments(segments, norm_grads)
  if (top_two_segments.size < 2):
    print("Couldn't find enough segments")
    return [], [], []

  # Update good_mask to only include top two groups
  a_segment = segments == top_two_segments[0]
  b_segment = segments == top_two_segments[1]
  good_mask = a_segment | b_segment 

  a_segment_idxs = np.argwhere(a_segment).flatten()
  b_segment_idxs = np.argwhere(b_segment).flatten()

  # Rotate angles 45 degrees first
  angle_a = np.mean(angles[a_segment_idxs]) + np.pi/4
  dir_a = np.array([np.cos(angle_a), np.sin(angle_a)])
  angle_b = np.mean(angles[b_segment_idxs]) + np.pi/4
  dir_b = np.array([np.cos(angle_b), np.sin(angle_b)])

  # Plot intersections
  chess_pts = getAllLineIntersections(lines[a_segment_idxs], lines[b_segment_idxs])
  pruned_chess_pts = prunePoints(chess_pts,max_dist2=5**2)
  return lines[a_segment_idxs], lines[b_segment_idxs], pruned_chess_pts, (dir_a, dir_b)