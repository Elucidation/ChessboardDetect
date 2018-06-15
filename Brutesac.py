
# coding: utf-8

# We have a bunch of probably chess x-corners output by our ML detector. Let's do the following routine:
# 
# 1. Take 4 random points, (check no 3 points are colinear, else retry) then warp them to a unity grid.
# 2. Count distances of each point to closest integer grid corner, sum of these will be the score.
# 3. Do this for an exhaustive set of random points?
# 
# The best score will likely be the correct transform to a chessboard.

# In[731]:

import numpy as np
from IPython.display import Image, display
import PIL.Image
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2 # For Sobel etc
import glob
import RunExportedMLOnImage
from scipy.spatial import Delaunay
from functools import wraps
import time

def timed(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    start = time.time()
    result = f(*args, **kwds)
    elapsed = time.time() - start
    print "%s took %.2f ms to finish" % (f.__name__, elapsed*1e3)
    return result
  return wrapper

# # RANSAC to find the best 4 points signifying a chess tile.
# Idea to find quads from triangles. For each triangle, the triangle that completes the quad should always be on the longest edge of the triangle and shares the two vertices of that edge. No image gradient checks needed.

# Build up a list of quads from input delaunay triangles, returns an Nx4 list of indices on the points used.
def getAllQuads(tri):
    pairings = set()
    quads = []
    # Build up a list of all possible unique quads from triangle neighbor pairings. 
    # In general the worst common case with a fully visible board is 6*6*2=36 triangles, each with 3 neighbor
    # so around ~100 quads.
    for i, neighbors in enumerate(tri.neighbors):
        for k in range(3): # For each potential neighbor (3, one opposing each vertex of triangle)
            nk = neighbors[k]
            if nk != -1:
                # There is a neighbor, create a quad unless it already exists in set
                pair = (i, nk)
                reverse_pair = (nk, i)
                if reverse_pair not in pairings:
                    # New pair, add and create a quad
                    pairings.add(pair)
                    b = tri.simplices[i]
                    d = tri.simplices[nk]                
                    nk_vtx = (set(d) - set(b)).pop()
                    insert_mapping = [2,3,1]
                    b = np.insert(b,insert_mapping[k], nk_vtx)
                    quads.append(b)
    return np.array(quads)

def countHits(given_pts, x_offset, y_offset):
    # Check the given integer points (in unity grid coordinate space) for matches
    # to an ideal chess grid with given initial offsets
    pt_set = set((a,b) for a,b in given_pts)
    [X,Y] = np.meshgrid(np.arange(7) + x_offset,np.arange(7) + y_offset)
    matches = 0
    # count matching points in set
    matches = sum(1 for x,y in zip(X.flatten(), Y.flatten()) if (x,y) in pt_set)
    return matches
        
def getBestBoardMatchup(given_pts):
    best_score = 0
    best_offset = None
    # scores = np.zeros([7,7])
    for i in range(7):
        for j in range(7):
            # Offsets from -6 to 0 for both
            score = countHits(given_pts, i-6, j-6)
            # scores[i,j] = score
            if score > best_score:
                best_score = score
                best_offset = [i-6, j-6]
    # print scores
    return best_score, best_offset

def scoreQuad(quad, pts, prevBestScore=0):
    idealQuad = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), idealQuad)
    
    # warped_to_ideal = cv2.perspectiveTransform(np.expand_dims(quad.astype(float),0), M)

    # Warp points and score error
    pts_warped = cv2.perspectiveTransform(np.expand_dims(pts.astype(float),0), M)[0,:,:]
    
    # Get their closest idealized grid point
    pts_warped_int = pts_warped.round().astype(int)

    # Refine
    M_refined, _ = cv2.findHomography(pts, pts_warped_int, cv2.RANSAC)
    if (M_refined is None):
      M_refined = M

    # Re-Warp points with the refined M and score error
    pts_warped = cv2.perspectiveTransform(np.expand_dims(pts.astype(float),0), M_refined)[0,:,:]
    
    # Get their closest idealized grid point
    pts_warped_int = pts_warped.round().astype(int)

    
    # Count matches
    score, offset = getBestBoardMatchup(pts_warped_int)
    if score < prevBestScore:
        return score, None, None, None
    
    # Sum of distances from closest integer value for each point
    # Use this error score for tie-breakers where number of matches is the same.
    error_score = np.sum(np.linalg.norm((pts_warped - pts_warped_int), axis=1))
    
    return score, error_score, M, offset

# brutesacChessboard ~752ms
# getAllQuads 12ms

@timed
def brutesacChessboard(xcorner_pts):
  # Build a list of quads to try.
  tri = Delaunay(xcorner_pts)
  quads = getAllQuads(tri) #
  
  # For each quad, keep track of the best fitting chessboard.
  best_score = 0
  best_error_score = None
  best_M = None
  best_quad = None
  best_offset = None
  for quad in xcorner_pts[quads]:
    score, error_score, M, offset = scoreQuad(quad, xcorner_pts, best_score)
    if score > best_score or (score == best_score and error_score < best_error_score):
      best_score = score
      best_error_score = error_score
      best_M = M
      best_quad = quad
      best_offset = offset
      if best_score > (len(xcorner_pts)*0.8):
        break

  return best_M, best_quad, best_offset, best_score, best_error_score


# @timed
def refineHomography(pts, M, best_offset):
  pts_warped = cv2.perspectiveTransform(np.expand_dims(pts.astype(float),0), M)[0,:,:]
  a = pts_warped.round() - best_offset
  b = pts

  # Only use points from within chessboard region.
  outliers = np.any((a < 0) | (a > 7), axis=1)
  a = a[~outliers]
  b = b[~outliers]

  # Least-Median robust homography
  M_homog, _ = cv2.findHomography(b,a, cv2.LMEDS)

  return M_homog

@timed
def predictOnTiles(tiles, predict_fn):
  predictions = predict_fn(
    {"x": tiles})

  # Return array of probability of tile being an xcorner.
  # return np.array([p[1] for p in predictions['probabilities']])
  return np.array([p[1] for p in predictions['probabilities']])

@timed
def predictOnImage(pts, img_gray, predict_fn, WINSIZE = 10):
  # Build tiles to run classifier on. (23 ms)
  tiles = getTilesFromImage(pts, img_gray, WINSIZE=WINSIZE)

  # Classify tiles. (~137ms)
  probs = predictOnTiles(tiles, predict_fn)

  return probs

@timed
def getTilesFromImage(pts, img_gray, WINSIZE=10):
  # NOTE : Assumes no point is within WINSIZE of an edge!

  # Points Nx2, columns should be x and y, not r and c.
  # Build tiles
  img_shape = np.array([img_gray.shape[1], img_gray.shape[0]])
  tiles = np.zeros([len(pts), WINSIZE*2+1, WINSIZE*2+1])
  for i, pt in enumerate(pts):
    tiles[i,:,:] = img_gray[pt[1]-WINSIZE:pt[1]+WINSIZE+1, pt[0]-WINSIZE:pt[0]+WINSIZE+1]

  return tiles

@timed
def classifyPoints(pts, img_gray, predict_fn, WINSIZE = 10):
  tiles = getTilesFromImage(pts, img_gray, WINSIZE=WINSIZE)

  # Classify tiles.
  probs = predictOnTiles(tiles, predict_fn)

  return probs

@timed
def classifyImage(img_gray, predict_fn, WINSIZE = 10, prob_threshold=0.5):
  spts = RunExportedMLOnImage.getFinalSaddlePoints(img_gray, WINSIZE)

  return spts[predictOnImage(spts, img_gray, predict_fn, WINSIZE) > prob_threshold, :]

@timed
def processFrame(gray):
  # and return M for chessboard from image
  pts = classifyImage(gray)

  # pts = np.loadtxt('example_pts.txt')
  pts = pts[:,[1,0]] # Switch rows/cols to x/y for plotting on an image

  if (len(pts)) < 6:
      print("Probably not enough points (%d) for a good fit, skipping" % len(pts))
      return None, None

  raw_M, best_quad, best_offset, best_score, best_error_score = brutesacChessboard(pts)

  # The chess outline is only a rough guess, which we can easily refine now that we have an understanding of which points are meant for which chess corner. One could do distortion correction with the points available (as normal chessboard algorithms are used for in computer vision). If we wanted to get fancy we could apply a warp flow field on a per-tile basis.

  # ## Refine with Homography on all points
  # Now we can refine our best guess using all the found x-corner points to their ideal points with opencv findHomography, using the regular method with all points.
  if (raw_M is not None):
    M_homog = refineHomography(pts, raw_M, best_offset)
  else:
    M_homog = None

  return M_homog, pts



if __name__ == '__main__':
  np.set_printoptions(suppress=True) # Better printing of arrays
  plt.rcParams['image.cmap'] = 'jet' # Default colormap is jet

  # img = PIL.Image.open('frame900.png')
  # img = PIL.Image.open('input/img_35.jpg')
  filenames = glob.glob('input_yt/*.jpg')
  filenames.extend(glob.glob('input_yt/*.png'))
  filenames.extend(glob.glob('input/*.jpg'))
  filenames.extend(glob.glob('input/*.png'))
  # filenames.extend(glob.glob('input_bad/*.png'))
  # filenames.extend(glob.glob('input_bad/*.jpg'))
  filename = np.random.choice(filenames,1)[0]
  filename= 'weird.jpg'
  print(filename)
  img = PIL.Image.open(filename)
  if (img.size[0] > 800):
      img = img.resize([600,400])
  gray = np.array(img.convert('L'))

  M_homog, pts = processFrame(gray)

  if M_homog is None:
    print("Failed on image, not enough points")
    exit()

  pts_warped2 = cv2.perspectiveTransform(np.expand_dims(pts.astype(float),0), M_homog)[0,:,:]

  fig = plt.figure(figsize=(10,5))
  ax = fig.add_subplot(1, 2, 1)

  # Major ticks every 20, minor ticks every 5
  major_ticks = np.arange(-2, 9, 2)
  minor_ticks = np.arange(-2, 9, 1)

  ax.axis('square')
  ax.set_xticks(major_ticks)
  ax.set_xticks(minor_ticks, minor=True)
  ax.set_yticks(major_ticks)
  ax.set_yticks(minor_ticks, minor=True)

  # And a corresponding grid
  ax.grid(which='both')

  # Or if you want different settings for the grids:
  ax.grid(which='minor', alpha=0.2)
  ax.grid(which='major', alpha=0.5)

  ideal_grid_pts = np.vstack([np.array([0,0,1,1,0])*8-1, np.array([0,1,1,0,0])*8-1]).T

  plt.plot(ideal_grid_pts[:,0], ideal_grid_pts[:,1], 'r:')
  plt.plot(pts_warped2[:,0], pts_warped2[:,1], 'r.')

  plt.subplot(122)
  plt.imshow(img)
  plt.plot(pts[:,0], pts[:,1], 'ro')

  # Refined via homography of all valid points
  unwarped_ideal_chess_corners_homography = cv2.perspectiveTransform(
      np.expand_dims(ideal_grid_pts.astype(float),0), np.linalg.inv(M_homog))[0,:,:]

  plt.plot(unwarped_ideal_chess_corners_homography[:,0], unwarped_ideal_chess_corners_homography[:,1], 'r-', lw=4);
  plt.show()