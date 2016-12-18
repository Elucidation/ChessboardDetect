import PIL.Image
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2 # For Sobel etc
import numpy as np
from helpers import *
from line_intersection import *
from chess_detect_helper import *
from rectify_refine import *
import os
np.set_printoptions(suppress=True) # Better printing of arrays

SAVE_RECTIFIED = False # Save rectified images out
SAVE_PLOT = False # Save plots (doesn't need to visualize)
SHOW_PLOTS = True # Visualize plots

input_folder = "input2"
output_folder = "rectified"
plot_folder = "plots"


for i in [23]:
  filename ="%02d.jpg" % i
# for filename in os.listdir(input_folder):
  filepath = "%s/%s" % (input_folder,filename)
  output_filename = output_folder+"/"+filename[:-3]+"png"
  # if (os.path.exists(output_filename)):
  #   print("%s exists, skipping %s" % (output_filename, filename))
  #   continue

  print("Processing %s" % filename)
  img_orig = scaleImageIfNeeded(PIL.Image.open(filepath))

  # Grayscale
  img = np.array(img_orig.convert('L')) # grayscale uint8 numpy array

  # Local Histogram Equalization
  # TODO : Currently breaks line detection etc., 
  #        tuning should be optimized with this equalization at some point
  # img = cv2.equalizeHist(img)
  # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  # img = clahe.apply(img)

  ##################
  ## Find initial set of chess lines in image using hough lines & gradient pruning
  lines_a, lines_b, chess_pts, top_dirs = getChessLinesCorners(img, chessboard_to_screen_ratio = 0.2)
  if (len(chess_pts) < 4):
    lines_a, lines_b, chess_pts, top_dirs = getChessLinesCorners(img, chessboard_to_screen_ratio = 0.15)
  if (len(chess_pts) < 4):
    lines_a, lines_b, chess_pts, top_dirs = getChessLinesCorners(img, chessboard_to_screen_ratio = 0.3)
  if (len(chess_pts) == 0):
    print("Couldn't get result for %s, skipping" % filename)
    continue
  elif (len(chess_pts) < 4):
    print("Couldn't get enough chess points: ", lines_a, lines_b, chess_pts, top_dirs)
    continue
  chess_pts = chess_pts[np.argsort(chess_pts[:,0]),:] # Sort by y height (row)

  ################## 
  # Find initial guess for chessboard corners and generate rectified image
  corners = getCorners(chess_pts, top_dirs)

  # Find perspective transform between corners of image to an idealized overhead
  # We add on two tiles in each direction to account for potential missing lines
  #  (the assumption being the algorithm should be able to find lines within 2 of edge always)
  # Assume missing up to 4 tiles along an axis
  warped_img, M = getTileImage(img_orig, corners, tile_buffer=1+4, tile_res=66)
  M_inv = np.matrix(np.linalg.inv(M))

  ##################
  # Get full chessboard line set on rectified image
  lines_x, lines_y, step_x, step_y = getRectifiedChessLines(warped_img)
  if not len(lines_x) or not len(lines_y):
    print("%s : Skipping, not enough lines in warped image" % filename)
    continue

  # Get edges and internal chessboard corners on rectified image
  warp_corners, all_warp_corners = getRectChessCorners(lines_x, lines_y)

  # Transform from rectified points back to original points for visualization
  tile_centers = all_warp_corners + np.array([step_x/2.0, step_y/2.0]) # Offset from corner to tile centers
  real_corners, all_real_tile_centers = getOrigChessCorners(warp_corners, tile_centers, M_inv)   

  tile_res = 64 # Each tile has N pixels per side
  tile_buffer = 1
  better_warped_img, better_M = getTileImage(img_orig, real_corners, tile_buffer=tile_buffer, tile_res=tile_res)
  # _, better_M = getTileImage(img_orig, real_corners, tile_buffer=1+4, tile_res=66)

  # Further refine rectified image
  better_warped_img, was_rotated, refine_M = reRectifyImages(better_warped_img)
  # combined_M = better_M
  combined_M = np.matmul(refine_M,better_M)

  if was_rotated:
    print(" tile image was rotated")
  
  M_inv = np.matrix(np.linalg.inv(combined_M))
  
  # Get better_M based corners
  hlines = vlines = (np.arange(8)+tile_buffer)*tile_res
  hcorner = (np.array([0,8,8,0])+tile_buffer)*tile_res
  vcorner = (np.array([0,0,8,8])+tile_buffer)*tile_res
  ideal_corners = np.vstack([hcorner,vcorner]).T
  ideal_all_corners = np.array(list(itertools.product(hlines, vlines)))
  ideal_tile_centers = ideal_all_corners + np.array([tile_res/2.0, tile_res/2.0]) # Offset from corner to tile centers
  # Get refined real corners
  real_corners, all_real_tile_centers = getOrigChessCorners(ideal_corners, ideal_tile_centers, M_inv)
  # Get final refined rectified warped image for saving
  better_warped_img, _ = getTileImage(img_orig, real_corners, tile_buffer=tile_buffer, tile_res=tile_res)

  print("Final transform matrix from image to rectified:\n", combined_M)


  if SAVE_RECTIFIED:
    print(" Saving tile image to %s" % output_filename)
    PIL.Image.fromarray(better_warped_img).save(output_filename)

  if SHOW_PLOTS or SAVE_PLOT:
    ##################
    # Plot Top Left Image, initial corner finding setup
    fig = plt.figure(filename, figsize=(12,8))
    fig.subplots_adjust(left=0.05,right=.95,bottom=0.05,top=.95)
    plt.subplot(221,aspect='equal')
    plt.imshow(img_orig)

    # Lines
    for idx, line in enumerate(lines_a):
      x1, y1, x2, y2 = line
      plt.plot([x1,x2], [y1,y2],'b', lw=3, alpha=0.5)
      # plt.text(x1, y1-2,'%s' % idx, color='blue', size=8, alpha=0.5);
    for idx, line in enumerate(lines_b):
      x1, y1, x2, y2 = line
      plt.plot([x1,x2], [y1,y2],'g', lw=3, alpha=0.5)

    plt.plot(corners[[0,1,2,3,0],0], corners[[0,1,2,3,0],1], 'r', lw=5)

    plt.plot(chess_pts[:,0], chess_pts[:,1], 'ro',ms=3) # Points
    # for idx in range(chess_pts.shape[0]):
    #   plt.text(chess_pts[idx,0], chess_pts[idx,1]-2,'%d' % idx, color='red', size=8,);

    plt.title('Input chess board + overlay initial prediction')
    plt.axis([0,img_orig.size[0],img_orig.size[1], 0])

    ##################
    # Plot Top Right: Rectified image + lines
    plt.subplot(222,aspect='equal')
    plt.imshow(warped_img)

    # Overlay rectified lines
    for idx, x_pos in enumerate(lines_x):
      plt.plot([x_pos, x_pos], [min(lines_y), max(lines_y)], 'r', lw=4)
      # plt.text(x_pos, min(lines_y)-10,'%d' % idx, color='red', size=10);
    for idx, y_pos in enumerate(lines_y):
      plt.plot([min(lines_x), max(lines_x)], [y_pos, y_pos], 'g', lw=4)
      # plt.text(min(lines_x)-40, y_pos, '%d' % idx, color='green', size=10);
    plt.title('Rectified image and prediction pass #2')
    plt.axis([0,warped_img.shape[1],warped_img.shape[0], 0])
    
    ##################
    # Plot Bottom Left: Overlay original image
    plt.subplot(223,aspect='equal')
    plt.imshow(img_orig)

    # plt.plot(real_corners[:,0], real_corners[:,1], 'ro', ms=5)
    # plt.plot(corners[[0,1,2,3,0],0], corners[[0,1,2,3,0],1], 'b', lw=2)
    plt.plot(real_corners[[0,1,2,3,0],0], real_corners[[0,1,2,3,0],1], 'r', lw=7, alpha=0.75)
    plt.plot(all_real_tile_centers[:,0], all_real_tile_centers[:,1], 'gD-',ms=4,lw=2, alpha=0.75)
    # for i in range(all_real_tile_centers.shape[0]):
    #   plt.text(all_real_tile_centers[i,0], all_real_tile_centers[i,1], '%d' % i, color='white', size=8);

    plt.title('Overlay: Refined tile positions')
    plt.axis([0,img_orig.size[0],img_orig.size[1], 0])

    ##################
    # Plot Bottom Right: Updated tile map
    plt.subplot(224,aspect='equal')
    plt.imshow(better_warped_img)
    
    for i in range(1,8):
      ix = (i+tile_buffer)*tile_res
      iy0 = tile_buffer*tile_res
      plt.plot([ix, ix],
               [iy0,(8+tile_buffer)*tile_res],
               'r', lw=2)
      plt.text(ix-10, iy0-10, '%d' % i, color='white', size=10, fontweight='heavy');
    
    for i in range(1,8):
      iy = (i+tile_buffer)*tile_res
      ix0 = tile_buffer*tile_res
      plt.plot([ix0,(8+tile_buffer)*tile_res],
               [iy, iy],
               'g', lw=2)
      plt.text(ix0-25, iy+5, '%d' % i, color='white', size=10, fontweight='heavy');

    plt.title('Output refined tile map')
    plt.axis([0,better_warped_img.shape[1],better_warped_img.shape[0], 0])

    if SAVE_PLOT:
      output_plot_filename = plot_folder+"/"+filename[:-3]+"png"
      print(" Saving plot to %s" % output_plot_filename)
      plt.savefig(output_plot_filename, bbox_inches='tight')

print("Done")

if SHOW_PLOTS:
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