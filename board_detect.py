import cv2
import PIL.Image
import numpy as np
import sys
np.set_printoptions(suppress=True, precision=2)

def scaleImageIfNeeded(img, max_width=1024, max_height=1024):
  """Scale image down to max_width / max_height keeping aspect ratio if needed. Do nothing otherwise."""
  # Input and Output is a numpy array
  img = PIL.Image.fromarray(img)
  img_width, img_height = img.size
  print("Image size %dx%d" % (img_width, img_height))
  aspect_ratio = min(float(max_width)/img_width, float(max_height)/img_height)
  if aspect_ratio < 1.0:
    new_width, new_height = ((np.array(img.size) * aspect_ratio)).astype(int)
    print(" Resizing to %dx%d" % (new_width, new_height))
    return np.array(img.resize((new_width,new_height)))
  return np.array(img)

def getAngle(a,b,c):
  # Get angle given 3 side lengths, in degrees
  return np.arccos((a*a+b*b-c*c) / (2*a*b)) * 180.0 / np.pi

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
  # p - pr and q - qs.
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

def testTwoLineSegmentIntersection():
  print("Test Two Line Segment Intersection")

  a = np.array([0,0])
  b = np.array([0,2])
  c = np.array([1,0])
  d = np.array([-1,1])
  t = getTwoLineSegmentIntersection(a,b,c,d)
  print(t)

  print("Done")

def getSegmentTheta(line):
  x1,y1,x2,y2 = line
  theta = np.math.atan2(y2-y1, x2-x1)
  return theta

def getSquareness(cnt, perfect_square_threshold=0.96):
  # 4x2 array, rows are each point, columns are x and y
  center = cnt.sum(axis=0)/4

  # Side lengths of rectangular contour
  dd0 = np.sqrt(((cnt[0,:] - cnt[1,:])**2).sum())
  dd1 = np.sqrt(((cnt[1,:] - cnt[2,:])**2).sum())
  dd2 = np.sqrt(((cnt[2,:] - cnt[3,:])**2).sum())
  dd3 = np.sqrt(((cnt[3,:] - cnt[0,:])**2).sum())

  # diagonal ratio
  # xa = np.sqrt(((cnt[0,:] - cnt[2,:])**2).sum())
  # xb = np.sqrt(((cnt[1,:] - cnt[3,:])**2).sum())
  # xratio = xa/xb if xa < xb else xb/xa
  side_ratio = dd0/dd1 if dd0 < dd1 else dd1/dd0
  if side_ratio > perfect_square_threshold:
    side_ratio = 1.0
  return side_ratio

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

  # All internal angles are at least 45 degrees but less than X degrees
  good_angles = np.all(np.array([ta,tb,tc,td]) > 35) and np.all(np.array([ta,tb,tc,td]) < (140))


  # side ratios
  dda = dd0 / dd1
  ddb = dd1 / dd2

  ddc = dd0/dd2
  ddd = dd1/dd3

  # Return whether side ratios within certain ratio < epsilon
  return (abs(1.0 - dda) < eps and abs(1.0 - ddb) < eps and 
          abs(1.0 - ddc) < 0.5 and abs(1.0 - ddd) < 0.5 and 
          xratio > xratio_thresh and angle_sum == 360 and good_angles)

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

def testMinDist():
  print("Test min dist")

  a = np.array([0,0])
  b = np.array([0,1.3])
  c = np.array([1.3,0.4])
  print(np.sqrt(minimum_distance2(a,b,c)))

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
  
  new_corners = tile_corners.copy()
  for i in range(4):
    x = getTwoLineSegmentIntersection(
      best_lines[i,:2], best_lines[i,2:],
      best_lines[(i+1)%4,:2], best_lines[(i+1)%4,2:])
    # print(best_lines, x)
    # print(best_lines[i,:2], best_lines[i,2:], best_lines[(i+1)%4,:2], best_lines[(i+1)%4,2:])
    if any(x):
      new_corners[i,:] = x

  return new_corners, best_lines


def findPotentialTiles(img):
  # blur img
  # img = (1.2*img - 0.2*cv2.blur(img,(3,3))).astype(np.uint8)
  img = cv2.bilateralFilter(img,3, 25, 75)
  # img = cv2.medianBlur(img,3)
  thresh = 100
  edges_orig = cv2.Canny(img, thresh, thresh*2)

  # Morphological Gradient to get internal squares of canny edges. 
  # kernel = np.ones((5,5),np.uint8)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  edges = cv2.morphologyEx(edges_orig, cv2.MORPH_GRADIENT, kernel)

  _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  contours = np.array(contours) # Turn to np array
  
  # Get dimmed image
  # img = (img.copy() * 0.9).astype(np.uint8)

  good_tiles = np.zeros(len(contours), dtype=bool)
  for i in range(contours.size):

    # Keep only internal contours (Has parent with findContour using cv2.RETR_CCOMP)
    if (hierarchy[0,i,3] < 0):
      # No parent found, skip outer contour
      continue

    # Approximate contour and update in place
    contours[i] = cv2.approxPolyDP(contours[i],0.02*cv2.arcLength(contours[i],True),True)
    
    # Only contours that fill an area of at least 8x8 pixels
    if cv2.contourArea(contours[i]) < 8*8:
      continue

    # Only rectangular contours allowed
    if len(contours[i]) != 4:
      continue

    # If rectangle is not square enough (even with leeway for perspective warp), remove
    if not is_square(contours[i][:,0,:]):
      continue

    # Survived tests, is good tile 
    good_tiles[i] = True
      
  # Prune bad contours
  contours = contours[good_tiles]

  # Calculate contour areas, then choose most common area
  areas = np.array(list(map(cv2.contourArea, contours)))
  
  # Sort contours by area size (largest first)
  area_max_order = np.argsort(areas)[::-1]
  contours = contours[area_max_order]
  areas = areas[area_max_order]

  med_area = np.median(areas)
  good_areas = np.abs(areas - med_area) < 0.5*med_area
  contours = contours[good_areas]


  # chosen_tile_idx = np.argsort(areas)[len(areas)//2]
  squareness_list = list(map(getSquareness, contours))

  # Sort contours by order of most square
  contours = contours[np.argsort(squareness_list)[::-1]]

  # Now contours are sorted by most square and largest area first

  return contours, 0, edges_orig

def getChosenTile(contours, chosen_tile_idx):
  return contours[chosen_tile_idx][:,0,:].astype(np.float32)

def drawPotentialTiles(img, contours, chosen_tile_idx):
  tile_corners = getChosenTile(contours, chosen_tile_idx)

  # Draw contours
  font = cv2.FONT_HERSHEY_PLAIN
  for i, cnt in enumerate(contours):
    if i == chosen_tile_idx:
      cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    else:
      cv2.drawContours(img,[cnt],0,(0,0,255),-1)


  cv2.line(img, tuple(tile_corners[0,:]), tuple(tile_corners[1,:]), (0,0,180), thickness=2)
  cv2.line(img, tuple(tile_corners[1,:]), tuple(tile_corners[2,:]), (0,180,0), thickness=2)
  cv2.line(img, tuple(tile_corners[2,:]), tuple(tile_corners[3,:]), (180,0,0), thickness=2)
  cv2.line(img, tuple(tile_corners[3,:]), tuple(tile_corners[0,:]), (0,0,0), thickness=2)

  cv2.putText(img,'0', tuple(contours[chosen_tile_idx][0,0,:]-5), font, 0.8,(0,0,0), thickness=1)
  cv2.putText(img,'1', tuple(contours[chosen_tile_idx][1,0,:]-5), font, 0.8,(0,0,0), thickness=1)
  cv2.putText(img,'2', tuple(contours[chosen_tile_idx][2,0,:]-5), font, 0.8,(0,0,0), thickness=1)
  cv2.putText(img,'3', tuple(contours[chosen_tile_idx][3,0,:]-5), font, 0.8,(0,0,0), thickness=1)


def drawSquareness(img, contours):
  squareness_list = np.array(list(map(getSquareness, contours)))
  # print(squareness_list)
  font = cv2.FONT_HERSHEY_PLAIN
  for i, cnt in enumerate(contours):    
    cv2.putText(img,'%.2f'%squareness_list[i], tuple(contours[i][0,0,:]-5), font, 0.5,(0,0,0), thickness=1)



def refineTile(img, edges, contours, chosen_tile_idx):
  tile_corners = getChosenTile(contours, chosen_tile_idx)
  
  tile_size = tile_corners.max(axis=0) - tile_corners.min(axis=0)
  tile_center = tile_corners.mean(axis=0)

  bbox_size_ratio = 4
  roi_bbox = np.hstack([tile_center-tile_size*bbox_size_ratio,tile_center+tile_size*bbox_size_ratio]).astype(int)
  
  # clamp bbox to img edges
  r,c,_ = img.shape
  roi_bbox[roi_bbox<0]=0
  roi_bbox[roi_bbox>[c,r,c,r]]= np.array([c,r,c,r])[roi_bbox>[c,r,c,r]]
  cv2.rectangle(img,tuple(roi_bbox[:2]),tuple(roi_bbox[2:]),(0,255,0),3)

  edges_roi = edges[ roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2] ]

  tile_side = int(tile_size.min())
  lines_roi = cv2.HoughLinesP(edges_roi,1,np.pi/180.0, tile_side, minLineLength=tile_side, maxLineGap=tile_side)
  
  if not np.any(lines_roi):
    print("No lines found")
    return
  
  lines_roi = lines_roi[:,0,:]
  # for line in lines_roi:
  #   line = (line + roi_bbox[[0,1,0,1]]).astype(np.int)

  # print("---")
  hough_lines = np.add(lines_roi, roi_bbox[[0,1,0,1]])
  hough_corners, corner_hough_lines = getBestCorners(tile_corners, hough_lines)
  return hough_corners, corner_hough_lines, edges_roi

def drawBestHoughLines(img, hough_corners, corner_hough_lines):
  # print(hough_corners)
  # print(corner_hough_lines)
  for line in corner_hough_lines:
    cv2.line(img, tuple(line[:2].astype(np.int)), tuple(line[2:].astype(np.int)), (255,255,255), thickness=2)

  for i in range(4):
    cv2.circle(img, 
      tuple(hough_corners[i,:]), 1, (0,0,0),thickness=-1)
  # print("---")

  # Draw 2x-chessboard expanded tile using simplistic multiplier instead of perspective transform
  # hough_tile_center = hough_corners.mean(axis=0)
  # expanded_tile_corners = hough_tile_center + (hough_corners - hough_tile_center)*(16+4)
  # cv2.polylines(img, [expanded_tile_corners.astype(np.int32)], True, (150,50,255), thickness=2)

  # -8 to 8
  # Single tile warp
  # M = cv2.getPerspectiveTransform(hough_corners,
  #                                 (tile_res)*(ideal_tile+8+1))
  # expanded tile area warp
  # M = cv2.getPerspectiveTransform(expanded_tile_corners,
  #                                 (tile_res)*(ideal_tile*tile_buffer))

  # print(M)
  # side_len = tile_res*(tile_buffer)
  # side_len = tile_res*(8 + 1 + tile_buffer)
  # out_img = cv2.warpPerspective(img, M,
  #                               (side_len, side_len))


def main(filenames):
  for filename in filenames:
    img = cv2.imread(filename)
    img = scaleImageIfNeeded(img)

    contours, chosen_tile_idx, edges = findPotentialTiles(img)
    drawPotentialTiles(img, contours, chosen_tile_idx)

    tile_corners = getChosenTile(contours, chosen_tile_idx)

    hough_corners, corner_hough_lines, edges_roi = refineTile(img, edges, contours, chosen_tile_idx)
    drawBestHoughLines(img, hough_corners, corner_hough_lines)

    # # Single tile warp
    # tile_res=64
    # M = cv2.getPerspectiveTransform(hough_corners,
    #                                 (tile_res)*(ideal_tile+8+1))
    # side_len = tile_res*(8 + 1 + tile_buffer)
    # out_img = cv2.warpPerspective(img, M,
    #                               (side_len, side_len))

    drawSquareness(img, contours)

    
    if img.size < 1000*1000:
      img = cv2.resize(img,None,fx=2, fy=2)
      edges_roi = cv2.resize(edges_roi,None,fx=2, fy=2)
    cv2.imshow(filename,img)
    cv2.imshow('edges',edges_roi)
    # cv2.imshow('%s_warped' % filename,out_img)
    # cv2.imshow('ROI',edges_roi)



  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  if len(sys.argv) > 1:
    filenames = sys.argv[1:]
  else:
    filenames = ['input/2.jpg']
  print("Loading", filenames)
  main(filenames)