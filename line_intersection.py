import numpy as np
from itertools import product

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
  pairings = np.array(list(product(range(linesA.shape[0]),range(linesB.shape[0]))))
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