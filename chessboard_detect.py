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
# for i in [10]:
# for i in [4,12,11,7,8]:
# for i in [7]:
# for i in [1,7,9]:
  filename = "%d.jpg" % i
  print("Processing %s" % filename)
  img_orig = scaleImageIfNeeded(PIL.Image.open(filename))

  # Grayscale
  img = np.array(img_orig.convert('L')) # grayscale uint8 numpy array


  lines_a, lines_b, chess_pts, top_dirs = getChessLinesCorners(img)
  if (len(chess_pts) == 0):
    print("Couldn't get result for %s, skipping" % filename)
    continue
  
  chess_pts = chess_pts[np.argsort(chess_pts[:,0]),:] # Sort by y height
  # chess_pts = chess_pts[np.argsort(np.sum(chess_pts,1)),:] # Sort by y height

  plt.figure(filename, figsize=(8,8))

  plt.subplot(221)
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
  ideal_corners = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float32)*400+200
  M = cv2.getPerspectiveTransform(corners, ideal_corners)
  warped_img = cv2.warpPerspective(np.array(img_orig), M, (800, 800))

  plt.subplot(222)
  plt.imshow(warped_img)

  # TODO: Fix awkward conversion
  # Convert RGB numpy array to image, then to grayscale image, then back to numpy array
  warped_img_gray = np.array(PIL.Image.fromarray(warped_img).convert('L'))
  warped_img_gray = cv2.bilateralFilter(warped_img_gray,15,75,75)

  # Find gradients
  sobelx = cv2.Sobel(warped_img_gray,cv2.CV_64F,1,0,ksize=5)
  # sobelx[warped_img_gray==0] = 0
  sobely = cv2.Sobel(warped_img_gray,cv2.CV_64F,0,1,ksize=5)
  # sobely[warped_img_gray==0] = 0

  sobelx_pos = sobelx.copy()
  sobelx_pos[sobelx <= 0] = 0
  sobelx_neg = sobelx.copy()
  sobelx_neg[sobelx > 0] = 0

  sobely_pos = sobely.copy()
  sobely_pos[sobely <= 0] = 0
  sobely_neg = sobely.copy()
  sobely_neg[sobely > 0] = 0

  checker_x = np.sum(sobelx_pos, axis=0) * np.sum(-sobelx_neg, axis=0)
  checker_x = skeletonize_1d(checker_x)

  checker_y = np.sum(sobely_pos, axis=1) * np.sum(-sobely_neg, axis=1)
  checker_y = skeletonize_1d(checker_y)

  x_lines = np.argwhere(checker_x).flatten()
  y_lines = np.argwhere(checker_y).flatten()

  if len(x_lines) < 7 or len(y_lines) < 7:
    print("%s : Skipping, not enough lines in warped image" % filename)
    continue

  # Select set of 7 adjacent lines with max sum score
  x_scores = np.zeros(7)
  for i in range(0,x_lines.shape[0]-7+1):
    x_scores[i] = np.sum(checker_x[x_lines[i:i+7]])
  x_start = np.argmax(x_scores)
  strongest_x_lines = range(x_start,x_start+7)

  y_scores = np.zeros(7)
  for i in range(0,y_lines.shape[0]-7+1):
    y_scores[i] = np.sum(checker_y[y_lines[i:i+7]])
  y_start = np.argmax(y_scores)
  strongest_y_lines = range(y_start,y_start+7)

  # strongest_x_lines = np.argsort(checker_x[x_lines])[::-1][:7]
  # strongest_y_lines = np.argsort(checker_y[y_lines])[::-1][:7]
  
  plt.subplot(223)
  plt.plot(checker_x, '.-')
  # print(strongest_x_lines)
  # print(x_lines)
  for k in strongest_x_lines:
    idx = x_lines[k]
    plt.plot([idx, idx], [0, np.max(checker_x)], 'r',lw=2)
  
  plt.subplot(224)
  plt.plot(checker_y, '.-')
  for k in strongest_y_lines:
    idx = y_lines[k]
    plt.plot([idx, idx], [0, np.max(checker_y)], 'g',lw=2)
  # plt.imshow(sobelx)

  
  plt.subplot(222)
  for idx in x_lines[strongest_x_lines]:
    plt.plot([idx, idx], [0, warped_img_gray.shape[0]], 'r', lw=2)
  for idx in y_lines[strongest_y_lines]:
    plt.plot([0, warped_img_gray.shape[1]], [idx, idx], 'g', lw=2)

  plt.axis('equal')

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