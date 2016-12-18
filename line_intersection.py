import PIL.Image
import cv2
import numpy as np
import itertools

# Calculate intersections
def line_intersect(a1, a2, b1, b2):
  T = np.array([[0, -1], [1, 0]])
  da = np.atleast_2d(a2 - a1)
  db = np.atleast_2d(b2 - b1)
  dp = np.atleast_2d(a1 - b1)
  dap = np.dot(da, T)
  denom = np.sum(dap * db, axis=1)
  num = np.sum(dap * dp, axis=1)
  return np.atleast_2d(num / denom).T * db + b1

def getAllLineIntersections(linesA, linesB):
  # get all pairings of lines
  pairings = np.array(list(itertools.product(range(linesA.shape[0]),range(linesB.shape[0]))))
  return line_intersect(linesA[pairings[:,0],:2], linesA[pairings[:,0],2:], linesB[pairings[:,1],:2], linesB[pairings[:,1],2:])

def prunePoints(pts, max_dist2=5**2):
  # Prune points away that are close to each other
  # preferring points that come earlier in array order
  good_pts = np.ones(pts.shape[0], dtype=bool)
  for i in range(pts.shape[0]):
    if ~good_pts[i]:
      continue
    for j in range(i+1,pts.shape[0]):
      d2 = np.sum((pts[j] - pts[i])**2)
      if (d2 < max_dist2): # within (N pixels)**2 of another point
        good_pts[j] = False

  return pts[good_pts]


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  a = np.array(
  [[265, 192, 517, 389],
   [210, 219, 427, 418],
   [352, 164, 594, 318],
   [254, 219, 459, 391],
   [295, 182, 544, 363],
   [330, 176, 570, 341],
   [360, 142, 617, 297],
   [332, 178, 541, 322],
   [295, 183, 505, 336],
   [252, 217, 488, 415],
   [360, 168, 567, 300],
   [289, 291, 455, 443],
   [232, 240, 454, 444],
   [286, 209, 478, 359]])
  b = np.array(
  [[253, 348, 605, 120],
   [274, 374, 604, 148],
   [318, 386, 558, 212],
   [230, 326, 578, 112],
   [348, 413, 650, 181],
   [207, 275, 440, 146],
   [209, 304, 531, 118],
   [319, 387, 560, 212],
   [256, 311, 579, 113],
   [260, 345, 606, 120],
   [191, 285, 387, 176],
   [234, 289, 530, 118],
   [210, 305, 531, 119],
   [275, 375, 604, 149]])
  
  pts = getAllLineIntersections(a, b)
  print("Found %d points" % len(pts))

  # Plot lines
  for line in a:
    x1, y1, x2, y2 = line
    plt.plot([x1,x2], [y1,y2],'b')
  for line in b:
    x1, y1, x2, y2 = line
    plt.plot([x1,x2], [y1,y2],'g')
  
  # Plot points
  plt.plot(pts[:,0], pts[:,1], 'ro',ms=8)

  plt.show()

def getCorners(chess_pts, top_dirs):
  """top_dirs are the two top direction vectors for the chess board lines"""
  d_norm_a = top_dirs[0]
  vals = chess_pts.dot(d_norm_a)
  a = chess_pts[np.argmin(vals),:]
  b = chess_pts[np.argmax(vals),:]

  dist = (b-a)
  d_norm = np.array([-dist[1], dist[0]])
  d_norm /= np.sqrt(np.sum(d_norm**2))

  # print(d_norm)
  vals = chess_pts.dot(d_norm)
  # print(vals)
  c = chess_pts[np.argmin(vals),:]
  d = chess_pts[np.argmax(vals),:]

  corners = np.vstack([a,c,b,d]).astype(np.float32)
  return corners

def getRectifiedChessLines(img):
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

    checker_x = np.sum(sobelx_pos, axis=0) * np.sum(-sobelx_neg, axis=0)
    checker_x = skeletonize_1d(checker_x)

    checker_y = np.sum(sobely_pos, axis=1) * np.sum(-sobely_neg, axis=1)
    checker_y = skeletonize_1d(checker_y)

    x_lines = np.argwhere(checker_x).flatten()
    y_lines = np.argwhere(checker_y).flatten()

    x_diff = np.diff(x_lines)
    y_diff = np.diff(y_lines)

    step_x_pred = np.median(x_diff)
    step_y_pred = np.median(y_diff)
    
    # Remove internal outlier lines that have the wrong step size
    x_good = np.ones(x_lines.shape,dtype=bool)
    y_good = np.ones(y_lines.shape,dtype=bool)
    
    x_good[1:] = abs(x_diff - step_x_pred) < 20
    y_good[1:] = abs(y_diff - step_y_pred) < 20

    x_keep = np.ones(x_lines.shape,dtype=bool)
    y_keep = np.ones(y_lines.shape,dtype=bool)

    for i in range(x_good.size-1):
      if ~np.any(x_good[i:i+2]):
        x_keep[i] = False
    for i in range(y_good.size-1):
      if ~np.any(y_good[i:i+2]):
        y_keep[i] = False

    x_lines = x_lines[x_keep]
    y_lines = y_lines[y_keep]

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

def skeletonize_1d(arr, win=50):
  """return skeletonized 1d array (thin to single value, favor to the right)"""
  _arr = arr.copy() # create a copy of array to modify without destroying original
  # Go forwards
  for i in range(_arr.size-1):
    if _arr[i] == 0:
      continue
    # Will right-shift if they are the same
    if np.any(arr[i] <= arr[i+1:i+win+1]):
      _arr[i] = 0
  
  # Go reverse
  for i in np.arange(_arr.size-1, 0,-1):
    if _arr[i] == 0:
      continue

    if np.any(arr[max(0,i-win):i] > arr[i]):
      _arr[i] = 0
  return _arr

def getRectChessCorners(lines_x, lines_y):
  pairs = np.array(list(itertools.product(range(8),range(8))))
  all_warp_corners = np.vstack([lines_x[pairs[:,0]], lines_y[pairs[:,1]]]).T
  warp_corners = np.array([
    [lines_x[0], lines_y[0]],
    [lines_x[-1], lines_y[0]],
    [lines_x[-1], lines_y[-1]],
    [lines_x[0], lines_y[-1]]
    ])
  return warp_corners[:,:2].astype(np.float32), all_warp_corners[:,:2].astype(np.float32)

def getOrigChessCorners(warp_corners, all_warp_corners, M_inv):
  all_stack = np.hstack([all_warp_corners, np.ones([all_warp_corners.shape[0],1])]).T
  all_real_corners = (M_inv * all_stack).T
  all_real_corners = all_real_corners / all_real_corners[:,2]

  stack = np.hstack([warp_corners, np.ones([4,1])]).T
  real_corners = (M_inv * stack).T
  real_corners = real_corners / real_corners[:,2] # Normalize by z
  return real_corners[:,:2].astype(np.float32), all_real_corners[:,:2].astype(np.float32)

def getTileImage(input_img, quad_corners, tile_buffer=0, tile_res=64):
  # Add N tile worth buffer on outside edge, such that
  # CV/ML algorithms could potentially use this data for better predictions
  ideal_quad_corners = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float32)

  main_len = tile_res*(ideal_quad_corners*8+tile_buffer)
  side_len = tile_res*(8+2*tile_buffer)

  M = cv2.getPerspectiveTransform(quad_corners, main_len)
  out_img = cv2.warpPerspective(np.array(input_img), M,
                                (side_len, side_len))
  return out_img, M

def getTileTransform(quad_corners, tile_buffer=0, tile_res=64):
  # Add N tile worth buffer on outside edge, such that
  # CV/ML algorithms could potentially use this data for better predictions
  ideal_quad_corners = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float32)

  main_len = tile_res*(ideal_quad_corners*8+tile_buffer)
  side_len = tile_res*(8+2*tile_buffer)

  M = cv2.getPerspectiveTransform(quad_corners, main_len)
  return M

def getSegments(v, eps = 2):
  # Get segment mask given a vector v, segments are values
  # withing eps distance of each other
  n = len(v)
  segment_mask = np.zeros(n,dtype=np.uint16)
  k = 1
  for i in range(n):
    if segment_mask[i] != 0:
      continue
    segment_mask[i] = k
    for j in range(i+1,n):
      if abs(v[j] - v[i]) < eps:
        segment_mask[j] = k
    k += 1
  return segment_mask-1, k-1

def mergePairs(pairs):
  if len(pairs) == 1:
    return pairs[0]
  
  vals = pairs[0]
  for i in range(1,len(pairs)):
    v_end = vals[-1]
    next_idx = np.argwhere(pairs[i] == v_end)
    if len(next_idx) > 0:
      vals = np.hstack([vals[:-1], pairs[i,next_idx[0][0]:]])
  return vals

def getBestEqualSpacing(vals, min_pts=7, eps=4, std_min=2):
  assert(min_pts>3)
  # Finds all combinations of triplets of points in vals where
  # the standard deviation is less than std_min, then merges
  # them into longer equally spaced sets and returns
  # the one with the largest equal spacing that has at least n_pts
  n_pts = 3
  pairs = np.array([k for k in itertools.combinations(vals, n_pts) if np.std(np.diff(k)) < std_min and np.mean(np.diff(k)) > 8])

  spacings = np.array([np.mean(np.diff(k)) for k in pairs])
  segments, num_segments = getSegments(spacings, eps)
  best_spacing = []
  best_mean = 0
  for i in range(num_segments):
    merged = mergePairs(pairs[segments==i])
    spacing_mean = np.mean(np.diff(merged))

    # Keep the largest equally spaced set that has min number of points
    if len(merged) >= min_pts and spacing_mean > best_mean:
      best_spacing = merged
      best_mean = spacing_mean

  return best_spacing