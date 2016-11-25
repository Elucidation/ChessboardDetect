import PIL.Image
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2 # For Sobel etc
import numpy as np
from helpers import *
from line_intersection import *
from chess_detect_helper import *
np.set_printoptions(suppress=True) # Better printing of arrays

for i in range(1,13):
# for i in [4,12,11,7,8]:
# for i in [12]:

  filename = "%d.jpg" % i
  img_orig = scaleImageIfNeeded(PIL.Image.open(filename))

  # Grayscale
  img = np.array(img_orig.convert('L')) # grayscale uint8 numpy array


  lines_a, lines_b, chess_pts, top_dirs = getChessLinesCorners(img)
  if (len(chess_pts) == 0):
    print("Couldn't get result for %s, skipping" % filename)
    continue
  
  chess_pts = chess_pts[np.argsort(chess_pts[:,0]),:] # Sort by y height
  # chess_pts = chess_pts[np.argsort(np.sum(chess_pts,1)),:] # Sort by y height

  plt.figure(filename, figsize=(20,8))

  plt.subplot(121)
  plt.imshow(img_orig)

  # Lines
  for idx, line in enumerate(lines_a):
    x1, y1, x2, y2 = line
    plt.plot([x1,x2], [y1,y2],'b', lw=3, alpha=0.5)
    # plt.text(x1, y1-2,'%s' % idx, color='blue', size=8, alpha=0.5);
  for idx, line in enumerate(lines_b):
    x1, y1, x2, y2 = line
    plt.plot([x1,x2], [y1,y2],'g', lw=3, alpha=0.5)

  plt.plot(chess_pts[:,0], chess_pts[:,1], 'ro',ms=5) # Points
  for idx in range(chess_pts.shape[0]):
    plt.text(chess_pts[idx,0], chess_pts[idx,1]-2,'%d' % idx, color='red', size=8);

  # Find corners of set of points
  corners = getCorners(chess_pts, top_dirs)
  plt.plot(corners[:,0], corners[:,1], 'y', lw=5)

  # Find perspective transform between corners of image to an idealized overhead
  # We add on two tiles in each direction to account for potential missing lines
  #  (the assumption being the algorithm should be able to find lines within 2 of edge always)
  ideal_corners = np.array([[0,0], [0,1], [1,1], [1,0]], dtype=np.float32)*400+200
  M = cv2.getPerspectiveTransform(corners, ideal_corners)
  warped_img = cv2.warpPerspective(img, M, (800, 800))

  plt.subplot(122)
  plt.imshow(warped_img)

  # print(M)

plt.show()

######################

# filename = "%d.jpg" % 8
# img_orig = scaleImageIfNeeded(PIL.Image.open(filename))

# # Grayscale
# img = np.array(img_orig.convert('L')) # grayscale uint8 numpy array

# # Edges
# # edges = cv2.Canny(img,50,150,apertureSize = 3)
# edges = cv2.Canny(img,200,500,apertureSize = 3, L2gradient=False) # Better thresholds

# # Gradients
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# grad_mag = np.sqrt(sobelx**2+sobely**2)

# # Hough Lines Probabilistic

# chessboard_to_screen_ratio = 0.25
# min_chessboard_line_length = chessboard_to_screen_ratio * min(img.shape)
# # TODO: This varys based on the chessboard to screen ratio, for chessboards filling the screen, we want to hop further
# max_line_gap = min_chessboard_line_length / 8.0 * 1.5 # Can hop up to one missing square
# # line_threshold = int(min_chessboard_line_length * 0.5)
# print("Min Chessboard Line Length: %g" % min_chessboard_line_length)
# print("Max Line gap: %g" % max_line_gap)
# # print("Line threshold: %d" % line_threshold)

# lines = cv2.HoughLinesP(edges,1,np.pi/360.0, 30, minLineLength = min_chessboard_line_length, maxLineGap = max_line_gap)[:,0,:]
# print("Number of lines: %d" % len(lines))

# fig1 = plt.figure(figsize=(20,8))
# plt.subplot(121)
# freqs = np.zeros(lines.shape[0])
# good_lines = np.zeros(lines.shape[0])
# edge_ratios = np.zeros(lines.shape[0])
# norm_grads = np.zeros(lines.shape[0])
# for idx in range(lines.shape[0]):
#   if idx > 100:
#     break
#   line = lines[idx,:]
#   is_good, strongest_freq, normal_gradients, fft_result, edge_ratio, avg_normal_gradient = getLineGradients(line, sobelx, sobely, grad_mag)
#   freqs[idx] = strongest_freq
#   good_lines[idx] = is_good
#   edge_ratios[idx] = edge_ratio
#   norm_grads[idx] = avg_normal_gradient
#   if is_good:
#     # print(idx, strongest_freq)
#     plt.plot(normal_gradients + idx*2.5)
#     plt.plot([0,600], [idx*2.5, idx*2.5],'k:')
#     plt.text(600-20, idx*2.5 + 0.1,'%s' % idx, color='green', size=8);
#     # plt.text(600-300, idx*2.5 + 0.1,'freq: %s, edge: %.2f' % (strongest_freq, edge_ratio), color='green', size=8);
#   else:
#     plt.plot(normal_gradients + idx*2.5, 'k', alpha=0.25)
#     plt.plot([0,600], [idx*2.5, idx*2.5],'k:',alpha=0.25)
#     plt.text(600-20, idx*2.5 + 0.1,'%s' % idx, color='black', size=8);
#     # plt.text(600-300, idx*2.5 + 0.1,'freq: %s, edge: %.2f' % (strongest_freq, edge_ratio), color='black', size=8);

# print("Number of good lines: %d" % np.sum(good_lines))
# # Get angles and segment lines up
# angles = np.zeros(lines.shape[0])
# for idx in range(lines.shape[0]):
#   line = lines[idx,:]
#   angles[idx] = getSegmentAngle(line)

# segments = segmentAngles(angles, good_lines)

# top_two_segments = chooseBestSegments(segments, norm_grads)

# # Update good_mask to only include top two groups
# a_segment = segments == top_two_segments[0]
# b_segment = segments == top_two_segments[1]
# good_mask = a_segment | b_segment 

# a_segment_idxs = np.argwhere(a_segment).flatten()
# b_segment_idxs = np.argwhere(b_segment).flatten()

# # print("segments",segments)
# # print("top two", top_two_segments)
# # print("good", good_lines)
# # print("freq", freqs)
# # print("edge", edge_ratios)
# # print("angles", np.floor(angles*180/np.pi))

# # Plot image
# plt.subplot(122)
# plt.imshow(img_orig)
# # plt.imshow(edges)
# plt.axis('equal')

# colors = 'krgbykrcmykrgbykcmyk'

# for k in a_segment_idxs:
#     line = lines[k,:]
#     x1, y1, x2, y2 = line
#     plt.plot([x1,x2], [y1,y2],'%s' % colors[segments[k]], lw=2)
#     plt.text(x1, y1-2,'%s' % k, color='blue', size=8);

# for k in b_segment_idxs:
#     line = lines[k,:]
#     x1, y1, x2, y2 = line
#     plt.plot([x1,x2], [y1,y2],'%s' % colors[segments[k]], lw=2)
#     plt.text(x1, y1-2,'%s' % k, color='blue', size=8);



# for k, [is_good, [x1,y1,x2,y2]] in enumerate(zip(good_mask, lines)):
#   if ~is_good:
#     plt.plot([x1,x2],[y1,y2], 'c', alpha=0.25)
#     plt.text(x1, y1-2,'%s' % k, color='blue', size=8, alpha=0.5);

# # Plot intersections
# chess_pts = getAllLineIntersections(lines[a_segment_idxs], lines[b_segment_idxs])
# pruned_chess_pts = prunePoints(chess_pts,max_dist2=5**2)

# # plt.plot(pruned_chess_pts[:,0], pruned_chess_pts[:,1], 'go',ms=2)

# better_chess_pts = pruned_chess_pts.copy()
# criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
# # better_chess_pts = cv2.cornerSubPix(img, better_chess_pts.astype(np.float32), (4,4), (-1,-1), criteria)
# plt.plot(better_chess_pts[:,0], better_chess_pts[:,1], 'ro',ms=5)
# print("Have %d points" % better_chess_pts.shape[0])

# plt.show()