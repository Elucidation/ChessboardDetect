# coding=utf-8
import PIL.Image
import matplotlib.image as mpimg
import scipy.ndimage
import cv2 # For Sobel etc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import skvideo.io
from functools import wraps
np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays

from scipy.spatial import Delaunay

def timed(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    start = time.time()
    result = f(*args, **kwds)
    elapsed = time.time() - start
    print "%s took %.2f ms to finish" % (f.__name__, elapsed*1e3)
    return result
  return wrapper

# export_dir = 'ml/model/001/1521934334' # old
# export_dir = 'ml/model/002/1528405915' # newer (same dataset, but random image changes)
# export_dir = 'ml/model/003/1528406613' # newer still
# export_dir = 'ml/model/004/1528441286' # win 21x21, 95% accuracy
# export_dir = 'ml/model/005/1528489968' # win 21x21 96% accuracy

# export_dir = 'ml/model/006/1528565066' # win 21x21 97% accuracy
# predict_fn = predictor.from_saved_model(export_dir, signature_def_key='predict')

def getModel(export_dir='ml/model/006/1528565066'):
  from tensorflow.contrib import predictor
  return predictor.from_saved_model(export_dir, signature_def_key='predict')

# Saddle
def getSaddle(gray_img):
    img = gray_img#.astype(np.float64)
    gx = cv2.Sobel(img,cv2.CV_32F,1,0)
    gy = cv2.Sobel(img,cv2.CV_32F,0,1)
    gxx = cv2.Sobel(gx,cv2.CV_32F,1,0)
    gyy = cv2.Sobel(gy,cv2.CV_32F,0,1)
    gxy = cv2.Sobel(gx,cv2.CV_32F,0,1)
    
    # Inverse everything so positive equals more likely.
    S = -gxx*gyy + gxy**2

    # Calculate subpixel offsets
    denom = (gxx*gyy - gxy*gxy)
    sub_s = (gy*gxy - gx*gyy) / denom
    sub_t = (gx*gxy - gy*gxx) / denom
    return S, sub_s, sub_t

def fast_nonmax_sup(img, win=11):
  element = np.ones([win, win], np.uint8)
  img_dilate = cv2.dilate(img, element)
  peaks = cv2.compare(img, img_dilate, cv2.CMP_EQ)
  img[peaks == 0] = 0


# Deprecated for fast_nonmax_sup
def nonmax_sup(img, win=10):
  w, h = img.shape
#     img = cv2.blur(img, ksize=(5,5))
  img_sup = np.zeros_like(img, dtype=np.float64)
  for i,j in np.argwhere(img):
    # Get neigborhood
    ta=max(0,i-win)
    tb=min(w,i+win+1)
    tc=max(0,j-win)
    td=min(h,j+win+1)
    cell = img[ta:tb,tc:td]
    val = img[i,j]
    # if np.sum(cell.max() == cell) > 1:
    #     print(cell.argmax())
    if cell.max() == val:
      img_sup[i,j] = val
  return img_sup

def pruneSaddle(s, init=128):
    thresh = init
    score = (s>0).sum()
    while (score > 10000):
        thresh = thresh*2
        s[s<thresh] = 0
        score = (s>0).sum()


def loadImage(filepath):
    img_orig = PIL.Image.open(filepath).convert('RGB')
    img_width, img_height = img_orig.size

    # Resize
    aspect_ratio = min(500.0/img_width, 500.0/img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width,new_height), resample=PIL.Image.BILINEAR)
    gray_img = img.convert('L') # grayscale
    img = np.array(img)
    gray_img = np.array(gray_img)
    
    return img, gray_img

@timed
def getFinalSaddlePoints(img, WINSIZE=10): # 32ms -> 15ms
  img = cv2.blur(img, (3,3)) # Blur it (.5ms)
  saddle, sub_s, sub_t = getSaddle(img) # 6ms
  fast_nonmax_sup(saddle) # ~6ms
  saddle[saddle<10000]=0 # Hardcoded ~1ms
  sub_idxs = np.nonzero(saddle)
  idxs = np.argwhere(saddle).astype(np.float64)
  spts = idxs[:,[1,0]] # Return in x,y order instead or row-col
  spts = spts + np.array([sub_s[sub_idxs], sub_t[sub_idxs]]).transpose()
  # Remove those points near win_size edges
  spts = clipBoundingPoints(spts, img.shape, WINSIZE)
  return spts # returns in x,y column order

def clipBoundingPoints(pts, img_shape, WINSIZE=10): # ~100us
  # Points are given in x,y coords, not r,c of the image shape
  a = ~np.any(np.logical_or(pts <= WINSIZE, pts[:,[1,0]] >= np.array(img_shape)-WINSIZE), axis=1)
  return pts[a,:]

def removeOutlierSimplices(tri):
    dists = np.zeros([tri.nsimplex, 3])
    for i,triangle in enumerate(tri.points[tri.simplices]):
        # We want the distance of the edge opposite the vertex k, so r_k.
        r0 = (triangle[2,:] - triangle[1,:])
        r1 = (triangle[0,:] - triangle[2,:])
        r2 = (triangle[1,:] - triangle[0,:])
        dists[i,:] = np.linalg.norm(np.vstack([r0,r1,r2]), axis=1)
    # Threshold based on twice the smallest edge.
    threshold = 2*np.median(dists)
    # Find edges that are too long
    long_edges = dists > threshold
    long_edged_simplices = np.any(long_edges,axis=1)
    # Good simplices are those that don't contain any long edges
    good_simplices_mask = ~long_edged_simplices
#     good_simplices = tri.simplices[good_simplices_mask]
    return dists, good_simplices_mask

def findQuadSimplices(tri, dists, simplices_mask=None):
    vertex_idx_opposite_longest_edge = dists.argmax(axis=1)
    # The neighboring tri for each tri about the longest edge
    potential_neighbor = tri.neighbors[
        np.arange(tri.nsimplex),
        vertex_idx_opposite_longest_edge]
    good_neighbors = []
    for i,j in enumerate(potential_neighbor):
        if j == -1:
            # Skip those that don't have a neighbor
            continue
        # If these tris both agree that they're good neighbors, keep them.
        if (potential_neighbor[j] == i and i < j):
            if simplices_mask is not None:
                if simplices_mask[i]:
                    good_neighbors.append(i)
                if simplices_mask[j]:
                    good_neighbors.append(j)
            else:
                good_neighbors.extend([i,j])
    return good_neighbors

# def videostream(filename='carlsen_match.mp4', SAVE_FRAME=True):
#   # vidstream = skvideo.io.vread('VID_20170427_003836.mp4')
#   # vidstream = skvideo.io.vread('VID_20170109_183657.mp4')
#   print("Loading video %s" % filename)
#   # vidstream = skvideo.io.vread('output2.avi')
#   vidstream = skvideo.io.vread(filename)#, num_frames=1000)
#   # vidstream = skvideo.io.vread('output.avi')
#   print("Finished loading")
#   # vidstream = skvideo.io.vread(0)
#   print(vidstream.shape)

#   # ffmpeg -i vidstream_frames/ml_frame_%03d.jpg -c:v libx264 -vf "fps=25,format=yuv420p"  test.avi -y

#   output_folder = "%s_vidstream_frames" % (filename[:-4])
#   if not os.path.exists(output_folder):
#     os.mkdir(output_folder)


#   for i, frame in enumerate(vidstream):
#     print("Frame %d" % i)
#     # frame = cv2.resize(frame, (320,240), interpolation = cv2.INTER_CUBIC)

#     # Our operations on the frame come here
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     inlier_pts, outlier_pts, pred_pts, final_predictions, prediction_levels, tri, simplices_mask = processImage(gray)


#     for pt in inlier_pts:
#       cv2.circle(frame, tuple(pt[::-1]), 3, (0,255,0), -1)

#     for pt in outlier_pts:
#       cv2.circle(frame, tuple(pt[::-1]), 1, (0,0,255), -1)

#     # Draw triangle mesh
#     if tri is not None:
#       cv2.polylines(frame,
#         np.flip(inlier_pts[tri.simplices].astype(np.int32), axis=2),
#         isClosed=True, color=(255,0,0))
#       cv2.polylines(frame,
#         np.flip(inlier_pts[tri.simplices[simplices_mask]].astype(np.int32), axis=2),
#         isClosed=True, color=(0,255,0), thickness=3)

#     cv2.putText(frame, 'Frame %d' % i, (5,15), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255),0,cv2.LINE_AA)

#     # Display the resulting frame
#     cv2.imshow('frame',frame)
#     output_filepath = '%s/ml_frame_%03d.jpg' % (output_folder, i)
#     if SAVE_FRAME:
#       cv2.imwrite(output_filepath, frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#   # When everything done, release the capture
#   cv2.destroyAllWindows()

# def calculateOutliers(pts, threshold_mult = 2.5):
#   N = len(pts)
#   std = np.std(pts, axis=0)
#   ctr = np.mean(pts, axis=0)
#   return (np.any(np.abs(pts-ctr) > threshold_mult * std, axis=1))




