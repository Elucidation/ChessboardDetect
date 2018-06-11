# coding=utf-8
import os
import cv2
import PIL.Image
import skvideo.io
import numpy as np
import Brutesac
import RunExportedMLOnImage
from functools import wraps
import time
from scipy.spatial import ConvexHull


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

@Brutesac.timed
def calculateOnFrame(gray, old_pts=None, old_gray=None, minPointsForLK=10):
  # and return M for chessboard from image
  if old_pts is not None:
    # calculate optical flow
    pts, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, old_pts.astype(np.float32), None, **lk_params)
    # pts = np.round(pts).astype(np.int32)
    # only points that LK found
    valid = st[:,0] != 0
    # only points that moved less than a couple pixels
    valid[np.abs(pts-old_pts).max(axis=1) > 5] = 0

    # Only points that are considered ML points (with lots of forgiveness).
    # validChessPoints = Brutesac.classifyPoints(pts.astype(np.int32), gray) > 0.3
    # valid[~validChessPoints] = 0

    pts = pts[valid,:]
    if len(pts) > minPointsForLK:

      # Update the valid points to the closest saddle point if possible
      spts = RunExportedMLOnImage.getFinalSaddlePoints(gray)
      spts = spts[:,[1,0]]
      newpts = []
      for i, pt in enumerate(pts):
        d = np.sum((spts - pt)**2, axis=1)
        best_spt_idx = d.argmin()
        # best_spt_idx = d.max(axis=1).argmin()
        # print(pt, spts[best_spt_idx,:], score)
        if d[best_spt_idx] < 3**2:
          newpts.append(spts[best_spt_idx,:])
        elif d[best_spt_idx] < 10**2:
          newpts.append(pt)
      if not newpts:
        pts = np.zeros([0,2])
      else:
        pts = np.array(newpts)

      print("LK")
  
  if old_pts is None or pts.shape[0] < minPointsForLK:
    pts = Brutesac.classifyImage(gray)
    if len(pts) == 0:
      return pts, []
    # pts = np.loadtxt('example_pts.txt')
    pts = pts[:,[1,0]] # Switch rows/cols to x/y for plotting on an image
    print("CLASSIFY")

  # Get contours
  contours, hierarchy = getContours(gray, pts)

  # xcorner_map = np.zeros(gray.shape, dtype=np.uint8)
  # for pt in pts:
  #   cv2.circle(xcorner_map, tuple(pt), 5, 1, -1)

  contours, hierarchy = pruneContours(contours, hierarchy, pts)

  return np.array(pts), contours

# @Brutesac.timed
def simplifyContours(contours):
  for i in range(len(contours)):
    # Approximate contour and update in place
    contours[i] = cv2.approxPolyDP(contours[i],0.04*cv2.arcLength(contours[i],True),True)

def updateCorners(contour, pts):
    new_contour = contour.copy()
    for i in range(len(contour)):
      cc,rr = contour[i,0,:]
      r = np.all(np.abs(pts - [cc,rr]) < 4, axis=1)
      closest_xpt = np.argwhere(r)
      if len(closest_xpt) > 0:
        new_contour[i,0,:] = pts[closest_xpt[0]][0]
      else:
          return []
    return new_contour


# @Brutesac.timed
def pruneContours(contours, hierarchy, xpts):
  new_contours = []
  new_hierarchies = []
  for i in range(len(contours)):
    cnt = contours[i]
    h = hierarchy[i]
    
    # Must be child
    if h[2] != -1:
        continue
    
    # Only rectangular contours allowed
    if len(cnt) != 4:
      continue
        
    # Only contours that fill an area of at least 8x8 pixels
    if cv2.contourArea(cnt) < 8*8:
      continue

    # if not is_square(cnt):
    #   continue
    
    # TODO : Remove those where internal luma variance is greater than threshold
    
    cnt = updateCorners(cnt, xpts)
    # If not all saddle corners
    if len(cnt) != 4:
        continue

    new_contours.append(cnt)
    new_hierarchies.append(h)
  
  new_contours = np.array(new_contours)
  new_hierarchy = np.array(new_hierarchies)
  if len(new_contours) == 0:
    return new_contours, new_hierarchy
  
  # Prune contours below median area
  areas = [cv2.contourArea(c) for c in new_contours]
  mask = [areas >= np.median(areas)*0.25] and [areas <= np.median(areas)*2.0]
  new_contours = new_contours[mask]
  new_hierarchy = new_hierarchy[mask]
  return np.array(new_contours), np.array(new_hierarchy)

# @Brutesac.timed
def getContours(gray, pts, iters=10):
  edges = cv2.Canny(gray, 20, 250)

  # Mask edges to only those in convex hull of points (dilated)
  if len(pts) >= 3:
    xcorner_mask = np.zeros(gray.shape, dtype=np.uint8)
    hull = ConvexHull(pts)
    hull_pts = np.round(pts[hull.vertices]).astype(np.int32)
    xcorner_mask = cv2.fillConvexPoly(xcorner_mask, hull_pts, 255)
    # Dilate mask a bit
    element = np.ones([21, 21], np.uint8)
    xcorner_mask = cv2.dilate(xcorner_mask, element)

    edges = cv2.bitwise_and(edges,edges,mask = xcorner_mask)

  # Morphological Gradient to get internal squares of canny edges. 
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
  _, contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  
  simplifyContours(contours,)

  if (hierarchy is None):
    return np.array(contours), None
  
  return np.array(contours), hierarchy[0]

@Brutesac.timed
def contourSacChessboard(xcorner_pts, quads):  
  # For each quad, keep track of the best fitting chessboard.
  best_score = 0
  best_error_score = None
  best_M = None
  best_quad = None
  best_offset = None
  for quad in quads:
    score, error_score, M, offset = Brutesac.scoreQuad(quad, xcorner_pts, best_score)
    if score > best_score or (score == best_score and error_score < best_error_score):
      best_score = score
      best_error_score = error_score
      best_M = M
      best_quad = quad
      best_offset = offset
      if best_score > (len(xcorner_pts)*0.9):
        break

  return best_M, best_quad, best_offset, best_score, best_error_score



@Brutesac.timed
def processFrame(frame, gray):
  frame_orig = frame.copy()
  pts, contours = calculateOnFrame(gray, processFrame.prevBoardpts, processFrame.prevGray)

  raw_M, best_quad, best_offset, best_score, best_error_score = contourSacChessboard(pts, contours)
  if raw_M is not None:
    M_homog = Brutesac.refineHomography(pts, raw_M, best_offset)
  else:
    M_homog = None

  # Draw tiles found
  # cv2.drawContours(frame,contours,-1,(0,255,255),2)

  # Draw xcorner points
  for pt in pts.astype(np.int64):
    cv2.circle(frame, tuple(pt), 3, (0,0,255), -1)


  ideal_grid_pts = np.vstack([np.array([0,0,1,1,0])*8-1, np.array([0,1,1,0,0])*8-1]).T
  xx,yy = np.meshgrid(np.arange(7), np.arange(7))
  all_ideal_grid_pts = np.vstack([xx.flatten(), yy.flatten()]).T

  if M_homog is not None:
    M_ideal_to_real = np.linalg.inv(M_homog)
    # Refined via homography of all valid points
    unwarped_ideal_chess_corners = cv2.perspectiveTransform(
        np.expand_dims(ideal_grid_pts.astype(float),0), M_ideal_to_real)[0,:,:]

    # Before offset
    # cv2.polylines(frame, 
    #   [unwarped_ideal_chess_corners.astype(np.int32)], 
    #   isClosed=True, thickness=0, color=(0,0,55))

    # Get a rectified chessboard image
    aligned_chess_corners = getAlignedChessCorners(unwarped_ideal_chess_corners[:4,:])


    tile_size = 32
    warpFrameQuad = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
    warpFrameQuad = (warpFrameQuad*(8)+2)*tile_size
    warpM = cv2.getPerspectiveTransform(aligned_chess_corners.astype(np.float32), warpFrameQuad)

    warp_frame_gray = cv2.warpPerspective(gray, warpM, (tile_size*12,tile_size*12))


    # Rectify grayscale chessboard image and find best offset for where chessboard really is
    # This undos potential off-by-[1,2] errors from propogated Classify/LK mistakes.
    best_offset, _ = findBestBoardViaTiles(warp_frame_gray)
    best_offset = (best_offset[0]-2, best_offset[1]-2) # Center on 1
    # best_offset = (0,0)
    # print(best_offset)

    # Rebuild ideal grid points with offset
    ideal_grid_pts -= best_offset
    all_ideal_grid_pts -= best_offset

    unwarped_ideal_chess_corners = cv2.perspectiveTransform(
        np.expand_dims(ideal_grid_pts.astype(float),0), M_ideal_to_real)[0,:,:]
    aligned_chess_corners = getAlignedChessCorners(unwarped_ideal_chess_corners[:4,:])

    unwarped_all_chesspts = cv2.perspectiveTransform(
        np.expand_dims(all_ideal_grid_pts.astype(float),0), M_ideal_to_real)[0,:,:]
    warpM = cv2.getPerspectiveTransform(aligned_chess_corners.astype(np.float32), warpFrameQuad)

    cv2.polylines(frame, 
      [unwarped_ideal_chess_corners.astype(np.int32)], 
      isClosed=True, thickness=4, color=(0,0,255))

    # cv2.circle(frame, tuple(unwarped_ideal_chess_corners[0,:].astype(np.int32)), 3, (255,255,255), -1)
    cv2.circle(frame, tuple(unwarped_ideal_chess_corners[0,:].astype(np.int32)), 3, (255,255,255), -1)

    # Keep only points that are classified as chess corner points
    # validChessPoints = Brutesac.classifyPoints(unwarped_all_chesspts[:,[1,0]].astype(np.int32), gray)[:,[1,0]]
    # validChessPoints = Brutesac.classifyPoints(unwarped_all_chesspts.astype(np.int32), gray)
    validChessPoints = unwarped_all_chesspts

    # for chess_pt in validChessPoints.astype(np.int32):
    #   cv2.circle(frame, tuple(chess_pt), 2, (0,255,255), -1)

    # for unwarped_chess_pt in unwarped_all_chesspts.astype(np.int32):
    #   cv2.circle(frame, tuple(unwarped_chess_pt), 2, (0,255,255), -1)

    # for valid_chess_pt in validChessPoints.astype(np.int32):
    #   cv2.circle(frame, tuple(valid_chess_pt), 2, (0,255,0), -1)
    # cv2.polylines(frame, 
    #   [unwarped_all_chesspts.astype(np.int32)], 
    #   isClosed=False, thickness=1, color=(0,255,255))

    processFrame.prevBoardpts = validChessPoints
    processFrame.prevGray = gray.copy()

    warpFrame = cv2.warpPerspective(frame_orig, warpM, (tile_size*12,tile_size*12))

    # idealQuad = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
    # actual_chessboard_corners = cv2.perspectiveTransform(
    #     np.expand_dims(idealQuad.astype(float),0), M_ideal_to_real)[0,:,:]
    # M = cv2.getPerspectiveTransform(unwarped_ideal_chess_corners.astype(np.float32), idealQuad)
    # M_homog = np.linalg.inv(warpM)

  else:
    processFrame.prevBoardpts = None
    processFrame.prevGray = None
    warpFrame = None
    aligned_chess_corners = None

  return frame, warpFrame, aligned_chess_corners

processFrame.prevBoardpts = None
processFrame.prevGray = None

def sumpool(a, shape):
  # re-bins array summing up components.
  sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
  return a.reshape(sh).sum(-1).sum(1)

@Brutesac.timed
def findBestBoardViaTiles(warp_frame_img_gray, tile_px=32):
  tilesum = sumpool(warp_frame_img_gray, np.array(warp_frame_img_gray.shape)/tile_px).astype(np.int64)

  # Now try 8x8 subsquares of this 12x12 array and take the 
  # difference between the sum of white and sum of black tiles (abs)
  # and use that as a score.
  same_color_tile_mask = np.tile(np.eye(2,dtype=bool),[4,4])
  i,j = 0,0

  best_score = 0
  best_idx = (-1,-1)
  for i in range(5):
    for j in range(5):
      sx = np.sum(tilesum[i:i+8,j:j+8][same_color_tile_mask])
      sy = np.sum(tilesum[i:i+8,j:j+8][~same_color_tile_mask])
      score = np.abs(sx-sy)
      if score > best_score:
        best_score = score
        best_idx = (j, i)

  return best_idx, tilesum

def getAlignedChessCorners(unaligned_corners):
  # Rotate corners until first point is closest to the top-left of the image and return.
  d = np.sum(unaligned_corners**2, axis=1)
  best = d.argmax()
  aligned_corners = np.roll(unaligned_corners, -best, axis=0)
  # d2 = np.sum(aligned_corners**2, axis=1)
  return aligned_corners


def getWarpedChessboard(img, M, tile_px=32):
  # Given a the 4 points of a chessboard, get a warped image of just the chessboard

  # board_pts = np.vstack([
  #   np.array([0,0,1,1])*tile_px,
  #   np.array([0,1,1,0])*tile_px
  #   ]).T
  img_warp = cv2.warpPerspective(img, M, (8*tile_px, 8*tile_px))
  return img_warp



def videostream(filepath='carlsen_match.mp4', output_folder_prefix='', SAVE_FRAME=True, MAX_FRAME=None):
  print("Loading video %s" % filepath)
  # vidstream = skvideo.io.vread(filepath, num_frames=4000)
  # Load frame-by-frame
  vidstream = skvideo.io.vreader(filepath)
  print("Finished loading")
  # print(vidstream.shape)

  # ffmpeg -i vidstream_frames/ml_frame_%03d.jpg -c:v libx264 -vf "fps=25,format=yuv420p"  test.avi -y
  filename = os.path.basename(filepath)

  output_folder = "%s/%s_vidstream_frames" % (output_folder_prefix, filename[:-4])
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)

  # Set up pts.txt, first line is the video filename
  # Following lines is the frame number and the flattened M matrix for the chessboard
  output_filepath_pts = '%s/pts.txt' % (output_folder)
  with open(output_filepath_pts, 'w') as f:
    f.write('%s\n' % filepath)

  for i, frame in enumerate(vidstream):
    # if i < 300:
    #   continue
    if i == MAX_FRAME:
      break
    print("Frame %d" % i)
    # if (i%5!=0):
    #   continue
    
    # frame = cv2.resize(frame, (320,240), interpolation = cv2.INTER_CUBIC)

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # if i == 63:
    #   cv2.imwrite('weird.png', frame)
    #   break;

    a = time.time()
    frame, warpFrame, chessboard_corners = processFrame(frame, gray)
    t_proc = time.time() - a

    # Add frame counter
    cv2.putText(frame, 'Frame % 4d (Processed in % 6.1f ms)' % (i, t_proc*1e3), (5,15), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255),0)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if warpFrame is not None:
      cv2.imshow('warpFrame',warpFrame)
    output_filepath = '%s/ml_frame_%03d.jpg' % (output_folder, i)
    output_filepath_warp = '%s/ml_warp_frame_%03d.jpg' % (output_folder, i)

    if SAVE_FRAME:
      cv2.imwrite(output_filepath, frame)
      if warpFrame is None:
        cv2.imwrite(output_filepath_warp, np.zeros_like(frame))
      else:
        cv2.imwrite(output_filepath_warp, warpFrame)

      # Append line of frame index and chessboard_corners matrix
      if chessboard_corners is not None:
        with open(output_filepath_pts, 'a') as f:
          chessboard_corners_str = ','.join(map(str,chessboard_corners.flatten()))
          # M_str = M.tostring() # binary
          f.write(u'%d,%s\n' % (i, chessboard_corners_str))

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

  # When everything done, release the capture
  # cap.release()
  cv2.destroyAllWindows()


def main():
  # filenames = glob.glob('input_bad/*')
  # filenames = glob.glob('input/img_*') filenames = sorted(filenames)
  # n = len(filenames)
  # filename = filenames[0]
  # filename = 'input/img_01.jpg'
  filename = 'weird.jpg'
  filename = 'chess_out1.png'

  print ("Processing %s" % (filename))
  img = PIL.Image.open(filename).resize([600,400])
  # img = PIL.Image.open(filename)
  rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  gray = np.array(img.convert('L'))

  ###
  rgb = processFrame(rgb, gray)
  ###

  cv2.imshow('frame',rgb)
  cv2.waitKey()

  print('Finished')


if __name__ == '__main__':
  # main()
  # filename = 'output2.avi' # Slow low rez
  # filename = 'random1.mp4' # Long video wait for 1k frames or so
  # filename = 'match2.mp4' # difficult
  # filename = 'output.avi' # Hard low rez
  # filename = 'output.mp4' # Hard
  # filename = 'speedchess1.mp4' # Great example
  # filename = 'wgm_1.mp4' # Lots of motion blur, slow
  # filename = 'gm_magnus_1.mp4' # Hard lots of scene transitions and blurry (init state with all pieces in a row not so good).
  # filename = 'bro_1.mp4' # Little movement, easy.
  # filename = 'chess_beer.mp4' # Reasonably easy, some off-by-N errors
  # filename = 'john1.mp4' # Simple clean
  filename = 'john2.mp4' # Slight motion, clean but slow
  # filename = 'swivel.mp4' # Moving around a fancy gold board

  allfiles = ['output2.avi', 'random1.mp4', 'match2.mp4','output.avi','output.mp4',
    'speedchess1.mp4','wgm_1.mp4','gm_magnus_1.mp4',
    'bro_1.mp4','chess_beer.mp4','john1.mp4','john2.mp4','swivel.mp4']

  for filename in allfiles:
    fullpath = 'datasets/raw/videos/%s' % filename
    output_folder_prefix = 'results'
    processFrame.prevBoardpts = None
    processFrame.prevGray = None
    print('\n\n - ON %s\n\n' % fullpath)
    videostream(fullpath, output_folder_prefix, True, MAX_FRAME=1000)



