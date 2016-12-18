from __future__ import print_function
import cv2
import PIL.Image
import numpy as np
import scipy.stats
import sys
import itertools
from line_intersection import *
np.set_printoptions(suppress=True, precision=2)

def scaleImageIfNeeded(img, max_width=1024, max_height=1024):
  """Scale image down to max_width / max_height keeping aspect ratio if needed. Do nothing otherwise."""
  # Input and Output is a numpy array
  img = PIL.Image.fromarray(img)
  img_width, img_height = img.size
  # print("Image size %dx%d" % (img_width, img_height))
  aspect_ratio = min(float(max_width)/img_width, float(max_height)/img_height)
  if aspect_ratio < 1.0:
    new_width, new_height = ((np.array(img.size) * aspect_ratio)).astype(int)
    # print(" Resizing to %dx%d" % (new_width, new_height))
    return np.array(img.resize((new_width,new_height)))
  return np.array(img)

def getAngle(a,b,c):
  # Get angle given 3 side lengths, in degrees
  k = (a*a+b*b-c*c) / (2*a*b)
  # Handle floating point errors
  if (k < -1):
    k=-1
  elif k > 1:
    k=1
  return np.arccos(k) * 180.0 / np.pi

def angleCloseDeg(a, b, angle_threshold=10):
  d = np.abs(a - b)
  # Handle angles that are ~180 degrees apart
  return d <= angle_threshold or np.abs(d-180) <= angle_threshold

def getSegmentThetaRho(line):
  x1,y1,x2,y2 = line
  theta = np.math.atan2(y2-y1, x2-x1)
  m = np.tan(theta)
  # rho = np.abs(y1 + m*x1) / np.sqrt(m*m+1)
  rho = x1*np.cos(theta) + y1*np.sin(theta)
  return theta, rho

def getTwoLineSegmentIntersection(p,pr,q,qs):
  # Uses http://stackoverflow.com/a/565282/2574639
  # Given two line segments defined by sets of points
  # (p -> pr) and (q -> qs).
  # Return the intersection point between them
  # *assumes it always exists for our particular use-case*
  
  # Convert to floats
  p = p.astype(np.float32)
  pr = pr.astype(np.float32)
  q = q.astype(np.float32)
  qs = qs.astype(np.float32)
  r = pr-p
  s = qs-q
  # print(p, pr, r)
  # print(q, qs, s)
  rxs = np.cross(r, s)
  if rxs == 0:
    return [] # parallel
  t = np.cross((q - p), s) / rxs
  return p + t*r # intersect

def chooseRandomGoodQuad(lines_a, lines_b, median_contour):
  # Get random set of points
  # Redo until min side distance of random corners greater than a multiple 
  # of the median tile found from initial estimator.
  sides_tile = getSquareSides(median_contour)
  for i in range(50):
    corners = chooseRandomQuad(lines_a, lines_b)
    sides_quad = getSquareSides(corners)
    if (i < 5):
      tile_size_mult = 5
    elif (i < 10):
      tile_size_mult = 4
    elif (i < 20):
      tile_size_mult = 3
    elif (i < 30):
      tile_size_mult = 2
    else:
      tile_size_mult = 1
    
    if min(sides_quad) > min(sides_tile*tile_size_mult):
      return corners
  
  print('chooseRandomGoodQuad hit max iter: %d' % i)
  return corners

def chooseRandomQuad(lines_a, lines_b):
  # Return 1 random quad (4 points) by choosing
  # 2 lines from lines_a and 2 lines from lines_b 
  # and returning their intersections
  a = np.random.choice(range(len(lines_a)),2, replace=False)
  b = np.random.choice(range(len(lines_b)),2, replace=False)

  pairs = np.array([
    [a[0], b[0]],
    [a[0], b[1]],
    [a[1], b[1]],
    [a[1], b[0]],
    ])

  corners = np.zeros([4,2])
  for i in range(4):
    k1 = lines_a[pairs[i,0]]
    k2 = lines_b[pairs[i,1]]
    corners[i,:] = getTwoLineSegmentIntersection(k1[:2], k1[2:], k2[:2], k2[2:])
  return corners


def getSegmentTheta(line):
  x1,y1,x2,y2 = line
  theta = np.math.atan2(y2-y1, x2-x1)
  return theta

def is_square(cnt, eps=3.0, xratio_thresh = 0.5):
  # 4x2 array, rows are each point, columns are x and y
  center = cnt.sum(axis=0)/4

  # Side lengths of rectangular contour
  dd0 = np.sqrt(((cnt[0,:] - cnt[1,:])**2).sum())
  dd1 = np.sqrt(((cnt[1,:] - cnt[2,:])**2).sum())
  dd2 = np.sqrt(((cnt[2,:] - cnt[3,:])**2).sum())
  dd3 = np.sqrt(((cnt[3,:] - cnt[0,:])**2).sum())

  # diagonal ratio
  xa = np.sqrt(((cnt[0,:] - cnt[2,:])**2).sum())
  xb = np.sqrt(((cnt[1,:] - cnt[3,:])**2).sum())
  xratio = xa/xb if xa < xb else xb/xa

  # Check whether all points part of convex hull
  # ie. not this http://i.stack.imgur.com/I6yJY.png
  # all corner angles, angles are less than 180 deg, so not necessarily internal angles
  ta = getAngle(dd3, dd0, xb) 
  tb = getAngle(dd0, dd1, xa)
  tc = getAngle(dd1, dd2, xb)
  td = getAngle(dd2, dd3, xa)
  angle_sum = np.round(ta+tb+tc+td)

  is_convex = angle_sum == 360

  angles = np.array([ta,tb,tc,td])
  good_angles = np.all((angles > 40) & (angles < 140))


  # side ratios
  dda = dd0 / dd1
  if dda < 1:
    dda = 1. / dda
  ddb = dd1 / dd2
  if ddb < 1:
    ddb = 1. / ddb
  ddc = dd2 / dd3
  if ddc < 1:
    ddc = 1. / ddc
  ddd = dd3 / dd0
  if ddd < 1:
    ddd = 1. / ddd
  side_ratios = np.array([dda,ddb,ddc,ddd])
  good_side_ratios = np.all(side_ratios < eps)

  # Return whether side ratios within certain ratio < epsilon
  return (
    # abs(1.0 - dda) < eps and 
    # abs(1.0 - ddb) < eps and
    # xratio > xratio_thresh and 
    # good_side_ratios and
    # is_convex and
    good_angles)

def minimum_distance2(v, w, p):
  # Return squared min distance between point p and line segment vw
  # Via http://stackoverflow.com/a/1501725
  # Return minimum distance between line segment vw and point p
  l2 = np.sum((v - w)**2)  # i.e. |w-v|^2 -  avoid a sqrt
  if (l2 == 0.0):
    return np.sum((p - v)**2)   # v == w case
  # Consider the line extending the segment, parameterized as v + t (w - v).
  # We find projection of point p onto the line. 
  # It falls where t = [(p-v) . (w-v)] / |w-v|^2
  # We clamp t from [0,1] to handle points outside the segment vw.
  t = max(0, min(1, np.dot(p - v, w - v) / l2))
  projection = v + t * (w - v)  # Projection falls on the segment
  return np.sum((p - projection)**2)

def getMinLineAngleDistance(a0, a1):
  # Compare line angles (which can be 180 off from one another, or +- 180)
  v0 = abs(a1-a0)
  v1 = abs((a1+np.pi) - a0)
  v2 = abs(a1 - (a0+np.pi))
  return min([v0,v1,v2])

def getBestCorners(tile_corners, hough_lines, angle_threshold = 10*np.pi/180):
  # Given 4x2 imperfect tile corners and Nx4 line segments
  # Expects line segments and corner points to be in same cartesian space
  #
  # Find 4 best line segments that are best match to the tile corners
  # and return the corners based off of those line segments, and those line segments
  best_lines = np.zeros([4,4])
  for i in range(4):
    corner_theta = getSegmentTheta(tile_corners[[i,i,((i+1)%4),((i+1)%4)], [0,1,0,1]])
    corner_ctr_pt = (tile_corners[i,:] + tile_corners[((i+1)%4),:]) / 2

    best_d = 1e6
    for line in hough_lines:
      theta = getSegmentTheta(line)
      # If angle within 10 degrees
      # if abs(corner_theta - theta) < angle_threshold:
      if getMinLineAngleDistance(corner_theta, theta) < angle_threshold:
        d = minimum_distance2(line[:2], line[2:], corner_ctr_pt)
        if d < best_d:
          best_d = d
          best_lines[i,:] = line
  
  new_corners = np.zeros([4,2], dtype=np.float32)
  for i in range(4):
    x = getTwoLineSegmentIntersection(
      best_lines[i,:2], best_lines[i,2:],
      best_lines[(i+1)%4,:2], best_lines[(i+1)%4,2:])
    # print(best_lines, x)
    # print(best_lines[i,:2], best_lines[i,2:], best_lines[(i+1)%4,:2], best_lines[(i+1)%4,2:])
    new_corners[i,:] = x

  return new_corners, best_lines

def simplifyContours(contours):
  for i in range(len(contours)):
    # Approximate contour and update in place
    contours[i] = cv2.approxPolyDP(contours[i],0.04*cv2.arcLength(contours[i],True),True)

def pruneContours(contours):
  new_contours = []
  for i in range(len(contours)):
    cnt = contours[i]    
    # Only contours that fill an area of at least 8x8 pixels
    if cv2.contourArea(cnt) < 8*8:
      continue
    # Only rectangular contours allowed
    if len(cnt) != 4:
      continue

    if not is_square(cnt):
      continue

    new_contours.append(cnt)
  new_contours = np.array(new_contours)
  if len(new_contours) == 0:
    return new_contours, None
  
  norm_contours = new_contours[:,:,0,:] - new_contours[:,[0],0,:]
  median_contour = np.median(norm_contours, axis=0).astype(int)
  diff = np.sqrt(np.sum((norm_contours - median_contour)**2,axis=2))

  new_contours = new_contours[np.all(diff < 60, axis=1)]

  return np.array(new_contours), median_contour

def getSquareSides(cnt):
  # 4x2 array, rows are each point, columns are x and y
  center = cnt.sum(axis=0)/4

  # Side lengths of rectangular contour
  dd0 = np.sqrt(((cnt[0,:] - cnt[1,:])**2).sum())
  dd1 = np.sqrt(((cnt[1,:] - cnt[2,:])**2).sum())
  dd2 = np.sqrt(((cnt[2,:] - cnt[3,:])**2).sum())
  dd3 = np.sqrt(((cnt[3,:] - cnt[0,:])**2).sum())
  return np.array([dd0, dd1, dd2, dd3])


from time import time
def calculateMask(mask_shape, contours, iters=10):
  a = time()
  sum_mask = np.zeros(mask_shape, dtype=int)
  tmp_mask = np.zeros(mask_shape, dtype=int)
  for i, cnt in enumerate(contours):
    for i in np.linspace(5,23,iters):
    # for i in [3,5,7,9,11,13,15]:
      # Calculate oversized tile mask and add to sum
      # big_cnt = (cnt.mean(axis=0) + (cnt-cnt.mean(axis=0))*i).astype(int)
      cnt_center = cnt.mean(axis=0)
      big_cnt = (cnt*i + cnt_center*(1-i)).astype(int)
      tmp_mask[:] = 0 # Reset
      cv2.drawContours(tmp_mask,[big_cnt],0,1,-1) # Fill mask with 1's inside contour
      sum_mask += tmp_mask

  # Normalize mask to 0-1 range
  sum_mask = sum_mask.astype(float) / sum_mask.max()
  # print("Mask calc took %.4f seconds." % (time() - a))
  return sum_mask

def getContourThetas(contours):
  thetas = []
  for cnt in contours:
    cnt = cnt[:,0,:]
    if cnt[0,0] < cnt[1,0]:
      side0 = np.hstack([cnt[1,:],cnt[0,:]])
    else:
      side0 = np.hstack([cnt[0,:],cnt[1,:]])
    if cnt[1,0] < cnt[2,0]:
      side1 = np.hstack([cnt[2,:],cnt[1,:]])
    else:
      side1 = np.hstack([cnt[1,:],cnt[2,:]])
    if cnt[2,0] < cnt[3,0]:
      side2 = np.hstack([cnt[3,:],cnt[2,:]])
    else:
      side2 = np.hstack([cnt[2,:],cnt[3,:]])
    if cnt[3,0] < cnt[0,0]:
      side3 = np.hstack([cnt[0,:],cnt[3,:]])
    else:
      side3 = np.hstack([cnt[3,:],cnt[0,:]])
    theta0 = getSegmentTheta(side0)
    theta1 = getSegmentTheta(side1)
    theta2 = getSegmentTheta(side2)
    theta3 = getSegmentTheta(side3)
    thetas.extend([theta0,theta1,theta2,theta3])
  return np.array(thetas)

def getEstimatedChessboardMask(img, edges, iters=10):
  # Morphological Gradient to get internal squares of canny edges. 
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

  _, contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  # Approximate polygons of contours
  simplifyContours(contours)

  if len(contours) == 0:
    return np.ones(img.shape[:2], dtype=float), None, None, None

  # Prune contours to rectangular ones
  contours, median_contour = pruneContours(contours)

  if len(contours) == 0 or median_contour is None:
    return np.ones(img.shape[:2], dtype=float), None, None, None

  thetas = getContourThetas(contours)
    
  top_two_angles = calculateKDE(thetas)

  mask = calculateMask(edges_gradient.shape, contours, iters)

  min_area_rect = getMinAreaRect(mask)

  return mask, top_two_angles, min_area_rect, median_contour


def calculateKDE(thetas):
  thetas *= 180/np.pi
  thetas[thetas<0] += 180
  
  kde_func = scipy.stats.gaussian_kde(thetas)
  positions = np.linspace(-40,180+40,360)
  kde_res = kde_func(positions)

  left_half = np.diff(kde_res)
  right_half = np.diff(kde_res[::-1])
  
  f = kde_res.copy()
  f[1:][left_half<0] = 0
  f[:-1][right_half[::-1]<0] = 0

  peak_indices = np.argwhere(f).flatten()
  peak_angles = positions[peak_indices]

  order = np.argsort(kde_res[peak_indices])[::-1] # strongest to weakest peaks

  return peak_angles[order[:2]] # top two strongest angles in degrees

# def plotKDE(thetas):
#   thetas *= 180/np.pi
#   thetas[thetas<0] += 180
  
#   ax1 = plt.subplot(211)
#   plt.plot(thetas,np.zeros(thetas.shape),'.')
#   plt.hist(thetas,20)

#   plt.subplot(212, sharex=ax1)
#   kde_func = scipy.stats.gaussian_kde(thetas)
#   positions = np.linspace(-40,180+40,360)
#   kde_res = kde_func(positions)
#   plt.plot(positions, kde_res)

#   c = kde_res.copy()
#   left_half = np.diff(kde_res)
#   right_half = np.diff(kde_res[::-1])
  
#   f = c.copy()
#   f[1:][left_half<0] = 0
#   f[:-1][right_half[::-1]<0] = 0
#   peak_indices = np.argwhere(f).flatten()
#   print(peak_indices, positions[peak_indices])
#   peak_angles = positions[peak_indices]

#   plt.plot(peak_angles, kde_res[peak_indices],'go')
#   order = np.argsort(kde_res[peak_indices][::-1]) # strongest to weakest peaks
#   return peak_angles[order[:2]], peak_angles, kde_res[peak_indices] # two angles in degrees

def getHoughLines(edges, min_line_size=100):
  # Expects chessboard to take up over 50% of edge map
  # min_line_size = int(min(edges.shape)/8)
  lines = cv2.HoughLinesP(edges,1,np.pi/360.0, int(min_line_size),
    minLineLength = min_line_size, maxLineGap = min_line_size/2)

  if lines is None:
    return np.array([])

  return lines[:,0,:]

def getSegmentTheta(line):
  x1,y1,x2,y2 = line
  theta = np.math.atan2(y2-y1, x2-x1)
  return theta

def parseHoughLines(lines, top_two_angles, angle_threshold_deg=20):
  is_good = np.zeros(len(lines)) # 0 = bad, 1 = close to 1st angle, 2 = close to 2nd angle
  for i, line in enumerate(lines):
    theta = getSegmentTheta(line) * 180/np.pi # to degrees
    d1 = getMinLineAngleDistanceDeg(theta, top_two_angles[0])
    d2 = getMinLineAngleDistanceDeg(theta, top_two_angles[1])
    if (d1 < angle_threshold_deg):
      is_good[i] = 1
    elif (d2 < angle_threshold_deg):
      is_good[i] = 2
  lines_a = lines[is_good==1]
  lines_b = lines[is_good==2]
  return lines_a, lines_b


def getMinLineAngleDistance(a0, a1):
  # In radians
  # Compare line angles (which can be 180 off from one another, or +- 180)
  v0 = abs(a1-a0)
  v1 = abs((a1+np.pi) - a0)
  v2 = abs(a1 - (a0+np.pi))
  return min([v0,v1,v2])

def getMinLineAngleDistanceDeg(a0, a1):
  # In degrees
  # Compare line angles (which can be 180 off from one another, or +- 180)
  v0 = abs(a1-a0)
  v1 = abs((a1+180) - a0)
  v2 = abs(a1 - (a0+180))
  return min([v0,v1,v2])


def plotHoughLines(img, lines, color=(255,255,255), line_thickness=2):
  # colors = np.random.random([lines.shape[0],3])*255
  # colors = np.array([
  #     [20,20,20],
  #     [255,0,0],
  #     [0,255,0],
  #     [255,255,0],
  #     [0,0,255],
  #     [255,0,255],
  #     [0,255,255],
  #     [200,200,200],
  #     ], dtype=np.uint8)
  # Plot lines
  for i, line in enumerate(lines):
    # color = list(map(int,colors[i%len(colors)]))
    cv2.line(img,
      tuple(line[:2].astype(np.int)),
      tuple(line[2:].astype(np.int)), color, thickness=line_thickness)



def getMinAreaRect(mask):
  a = np.argwhere(mask.T>0.5)
  # rect = cv2.boundingRect(a)
  rect = cv2.minAreaRect(a)
  return rect

def drawMinAreaRect(img, rect, color=(0,255,255)):
  ctr = tuple(map(int,rect[0]))
  
  box = cv2.boxPoints(rect)
  box = np.int0(box)

  cv2.drawContours(img,[box],0,color,2)
  cv2.circle(img, ctr, 3, (255,0,0),-1)


def skeletonize_1d(data):
  c = data.copy()
  left_half = np.diff(data)
  right_half = np.diff(data[::-1])

  f = data.copy()
  f[1:][left_half<0] = 0
  f[:-1][right_half[::-1]<0] = 0
  return f

def getWarpCheckerLines(img):
  """Given a warped axis-aligned image of a chessboard, return internal line crossings"""
  # TODO: Fix awkward conversion
  # Convert RGB numpy array to image, then to grayscale image, then back to numpy array
  img_gray = np.array(PIL.Image.fromarray(img).convert('L'))
  img_gray = cv2.bilateralFilter(img_gray,15,75,75)

  # Find gradients
  sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=5)
  sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=5)

  sobelx_pos = sobelx.copy()
  sobelx_pos[sobelx <= 0] = 0
  sobelx_neg = sobelx.copy()
  sobelx_neg[sobelx > 0] = 0

  sobely_pos = sobely.copy()
  sobely_pos[sobely <= 0] = 0
  sobely_neg = sobely.copy()
  sobely_neg[sobely > 0] = 0

  kernel = np.array([ 0.  ,  0.  ,  0.04,  0.32,  0.88,  0.88,  0.32,  0.04,  0.  ,  0.  ])

  checker_x = np.sum(sobelx_pos, axis=0) * np.sum(-sobelx_neg, axis=0)
  checker_x = np.convolve(checker_x, kernel, 'same')
  checker_x = checker_x / checker_x.max()
  checker_x[checker_x<0.1] = 0
  checker_x = skeletonize_1d(checker_x)

  checker_y = np.sum(sobely_pos, axis=1) * np.sum(-sobely_neg, axis=1)
  checker_y = np.convolve(checker_y, kernel, 'same')
  checker_y = checker_y / checker_y.max()
  checker_y[checker_y<0.1] = 0
  checker_y = skeletonize_1d(checker_y)

  x_lines = np.argwhere(checker_x).flatten()
  y_lines = np.argwhere(checker_y).flatten()



  #######
  ## Semi-brute force approach, merge all combinations of 3 points 
  # with equal spacing under one standard deviation
  x_lines = getBestEqualSpacing(x_lines)
  y_lines = getBestEqualSpacing(y_lines)

  ###########

  if len(x_lines) < 7 or len(y_lines) < 7:
    return [], [], [], []

  # Select set of 7 adjacent lines with max sum score
  x_scores = np.zeros(x_lines.shape[0]-7+1)
  for i in range(0,x_lines.shape[0]-7+1):
    x_scores[i] = np.sum(checker_x[x_lines[i:i+7]])
  x_start = np.argmax(x_scores)
  strongest_x_lines = range(x_start,x_start+7)

  y_scores = np.zeros(y_lines.shape[0]-7+1)
  for i in range(0,y_lines.shape[0]-7+1):
    y_scores[i] = np.sum(checker_y[y_lines[i:i+7]])
  y_start = np.argmax(y_scores)
  strongest_y_lines = range(y_start,y_start+7)

  # TODO: Sanity check areas between lines for consistent color when choosing?

  # Choose best internal 7 chessboard lines
  lines_x = x_lines[strongest_x_lines]
  lines_y = y_lines[strongest_y_lines]

  # Add outer chessboard edges assuming consistent step size
  step_x = np.median(np.diff(lines_x))
  step_y = np.median(np.diff(lines_y))

  lines_x = np.hstack([lines_x[0]-step_x, lines_x, lines_x[-1]+step_x])
  lines_y = np.hstack([lines_y[0]-step_y, lines_y, lines_y[-1]+step_y])

  return lines_x, lines_y, step_x, step_y

  # x_lines = np.argwhere(checker_x).flatten()
  # y_lines = np.argwhere(checker_y).flatten()

  # x_diff = np.diff(x_lines)
  # y_diff = np.diff(y_lines)

  # step_x_pred = np.median(x_diff)
  # step_y_pred = np.median(y_diff)

def pruneGradLines(a, b, eta=10):
  # Remove values from vector 'a' that aren't close to values in vector b
  is_good = np.zeros(len(a),dtype=bool)
  for i,v in enumerate(a):
    if min(b-v) < eta:
      is_good[i] = True
  return a[is_good]
  












def main(filenames):
  for filename in filenames:
    img = cv2.imread(filename)
    img = scaleImageIfNeeded(img, 600, 480)

    # Edges
    edges = cv2.Canny(img, 100, 550)
    mask, _, _, _ = getEstimatedChessboardMask(img, edges, iters=10)

    img_masked_full = cv2.bitwise_and(img,img,mask = (mask > 0.5).astype(np.uint8))
    img_masked = cv2.addWeighted(img,0.2,img_masked_full,0.8,0)
    edges_masked = cv2.bitwise_and(edges,edges,mask = (mask > 0.5).astype(np.uint8))

    cv2.imshow('img %s' % filename,img_masked)
    cv2.imshow('edges %s' % filename, edges_masked)
    cv2.imshow('mask %s' % filename, mask)


  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  if len(sys.argv) > 1:
    filenames = sys.argv[1:]
  else:
    # filenames = ['input/1.jpg']
    filenames = ['input2/18.jpg']
  print("Loading", filenames)
  main(filenames)