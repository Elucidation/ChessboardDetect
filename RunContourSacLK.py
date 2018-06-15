# coding=utf-8
import os
from argparse import ArgumentParser
import cv2
import PIL.Image
import skvideo.io
import numpy as np
import Brutesac
import RunExportedMLOnImage
from functools import wraps
import time
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

@Brutesac.timed
def calculateOnFrame(gray, predict_fn, old_pts=None, old_gray=None, minPointsForLK=10, WINSIZE=10, probability_threshold=0.8):
  # and return M for chessboard from image

  # TODO: pass region mask to classifyImage to only search in a region

  spts = RunExportedMLOnImage.getFinalSaddlePoints(gray)

  # old_pts = None # For testing classify only without LK.
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
      if False:
        # Calculate x-pts for the current image
        # xpts = Brutesac.classifyImage(gray, predict_fn, WINSIZE=WINSIZE)
        probabilities = Brutesac.predictOnImage(spts, gray, predict_fn, WINSIZE=WINSIZE)
        xpts = spts[probabilities > probability_threshold,:]

        # Find closest xpts
        min_dists, min_dist_idx = cKDTree(xpts).query(pts, 1)
        
        keep_mask = min_dists < 4
        replace_mask = np.logical_and(min_dists > 1, min_dists < 4)

        # Update those xpts that are within the right range and throw out those
        # too far away
        pts[replace_mask,:] = xpts[min_dist_idx[replace_mask],:]
        pts = pts[keep_mask,:]
      else:
        # Update the valid points to the closest saddle point if possible
        min_dists, min_dist_idx = cKDTree(spts).query(pts, 1)
        
        keep_mask = min_dists < 4
        replace_mask = np.logical_and(min_dists > 1, min_dists < 4)

        pts[replace_mask,:] = spts[min_dist_idx[replace_mask],:]
        pts = pts[keep_mask,:]
      print("LK")
  else:
  # if old_pts is None or pts.shape[0] < minPointsForLK:
    # pts = Brutesac.classifyImage(gray, predict_fn, WINSIZE=WINSIZE)
    probabilities = Brutesac.predictOnImage(spts, gray, predict_fn, WINSIZE=WINSIZE)
    pts = spts[probabilities > probability_threshold,:]
    if len(pts) == 0:
      return pts, []
    print("CLASSIFY %d pts" % len(pts))

  # Get contours.
  contours, hierarchy = getContours(gray, pts)
  contours, hierarchy = pruneContours(contours, hierarchy, pts)

  return pts, contours

# @Brutesac.timed
def simplifyContours(contours):
  for i in range(len(contours)):
    # Approximate contour and update in place
    contours[i] = cv2.approxPolyDP(contours[i],0.04*cv2.arcLength(contours[i],True),True)

def updateCorners(contour, pts):
  # Expects pts in x,y form
  new_contour = contour.copy()
  for i in range(len(contour)):
    cc,rr = contour[i,0,:]
    r = np.all(np.abs(pts - [cc,rr]) < 5, axis=1)
    closest_xpt = np.argwhere(r)
    # if there's at least one successful match nearby.
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
def processFrame(frame, gray, predict_fn):
  frame_orig = frame.copy()
  pts, contours = calculateOnFrame(gray, predict_fn, processFrame.prevBoardpts, processFrame.prevGray,
    WINSIZE=7)

  raw_M, best_quad, best_offset, best_score, best_error_score = contourSacChessboard(pts, contours)
  if raw_M is not None:
    M_homog = Brutesac.refineHomography(pts, raw_M, best_offset)
  else:
    M_homog = None

  # Draw tiles found
  cv2.drawContours(frame,contours,-1,(0,255,0),1)

  # Draw xcorner points
  for pt in np.round(pts).astype(np.int64):
    # cv2.circle(frame, tuple(pt), 3, (0,0,255), -1)
    cv2.rectangle(frame, tuple(pt-1),tuple(pt+1), (0,0,255), -1)


  ideal_grid_pts = np.vstack([np.array([0,0,1,1,0])*8-1, np.array([0,1,1,0,0])*8-1]).T.astype(np.float32)
  xx,yy = np.meshgrid(np.arange(7), np.arange(7))
  all_ideal_grid_pts = np.vstack([xx.flatten(), yy.flatten()]).T.astype(np.float32)

  if M_homog is not None:
    M_ideal_to_real = np.linalg.inv(M_homog)
    # Refined via homography of all valid points
    unwarped_ideal_chess_corners = cv2.perspectiveTransform(
        np.expand_dims(ideal_grid_pts,0), M_ideal_to_real)[0,:,:]

    # Before offset
    # cv2.polylines(frame, 
    #   [np.round(unwarped_ideal_chess_corners).astype(np.int32)], 
    #   isClosed=True, thickness=0, color=(0,0,55))

    # Get a rectified chessboard image
    aligned_chess_corners = getAlignedChessCorners(unwarped_ideal_chess_corners[:4,:])


    tile_size = 32
    warpFrameQuad = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
    buffer_tiles = 2
    tiles_plus_buffer = 8+buffer_tiles*2
    warpFrameQuad = (warpFrameQuad*(8)+buffer_tiles)*tile_size
    warpM = cv2.getPerspectiveTransform(aligned_chess_corners.astype(np.float32), warpFrameQuad)

    warp_frame_gray = cv2.warpPerspective(gray, warpM, 
      (tile_size*(tiles_plus_buffer),tile_size*(tiles_plus_buffer)))


    # Rectify grayscale chessboard image and find best offset for where chessboard really is
    # This undos potential off-by-[1,2] errors from propogated Classify/LK mistakes.
    DO_BOARD_REALIGN = True
    # DO_BOARD_REALIGN = False
    if DO_BOARD_REALIGN:
      best_offset, tilesum, best_score = findBestBoardViaTiles(warp_frame_gray)
      best_offset = (best_offset[0]-buffer_tiles, best_offset[1]-buffer_tiles) # Center on 1
      # if best_offset[0] != 0 and best_offset[1] != 0:
      #   best_offset = (0,0)
      # print(best_offset)
      # print(best_score)
      # if best_score < 5000:
      #   best_offset = (0,0)  
    else:
      tilesum = None
      best_offset = (0,0) # For testing without affecting results.

    # Rebuild ideal grid points with offset
    ideal_grid_pts -= best_offset
    all_ideal_grid_pts -= best_offset

    unwarped_ideal_chess_corners = cv2.perspectiveTransform(
        np.expand_dims(ideal_grid_pts,0), M_ideal_to_real)[0,:,:]
    aligned_chess_corners = getAlignedChessCorners(unwarped_ideal_chess_corners[:4,:])

    unwarped_all_chesspts = cv2.perspectiveTransform(
        np.expand_dims(all_ideal_grid_pts,0), M_ideal_to_real)[0,:,:]
    warpM = cv2.getPerspectiveTransform(aligned_chess_corners.astype(np.float32), warpFrameQuad)

    cv2.polylines(frame, 
      [np.round(unwarped_ideal_chess_corners).astype(np.int32)], 
      isClosed=True, thickness=4, color=(0,0,255))

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

    if best_offset[0] == 0 and best_offset[1] == 0:
      processFrame.prevBoardpts = validChessPoints
    else:
      processFrame.prevBoardpts = None

    processFrame.prevGray = gray.copy()

    warpFrame = cv2.warpPerspective(frame_orig, warpM, 
      (tile_size*tiles_plus_buffer,tile_size*tiles_plus_buffer))

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
    tilesum = None

  return frame, warpFrame, aligned_chess_corners, tilesum

processFrame.prevBoardpts = None
processFrame.prevGray = None

def sumpool(a, shape):
  # re-bins array summing up components.
  sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
  return a.reshape(sh).sum(-1).sum(1)

def medianpool(a, shape):
  # re-bins array summing up components.
  sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
  return np.median(np.median(a.reshape(sh), axis=-1), axis=1).astype(np.int64)

def meanpool(a, shape):
  # re-bins array summing up components.
  sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
  return a.reshape(sh).mean(-1).mean(1).astype(np.int64)

@Brutesac.timed
def findBestBoardViaTiles(warp_frame_img_gray, tile_px=32):
  # TODO : Consider using standard deviation of color or something instead.

  # In the range 0 to 1024 (from 32*32 = 1024 when downscaling by 32)
  # tilesum = medianpool(warp_frame_img_gray.astype(np.int64), 
  #   np.array(warp_frame_img_gray.shape)/tile_px)-512
  special_gray = warp_frame_img_gray.astype(np.int64)
  special_gray[special_gray==0] = 127 # Make out-of-borders average zero mean.
  # special_gray = special_gray - np.median(special_gray) # semi-normalize
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  special_gray = clahe.apply(special_gray.astype(np.uint8)).astype(np.int64) - 127

  tilesum = meanpool(special_gray, 
    np.array(special_gray.shape)/tile_px)

  n_max = tilesum.shape[0]


  if False:
    # New idea, take every other tile and sum the diffs of
    # it to it's one on the right and it to the one below.
    tile_mask = np.tile(np.eye(2,dtype=bool),[4,4])

    best_score = 0
    best_idx = (-1,-1)
    for y in range(n_max-8+1):
      for x in range(n_max-8+1):
        subtile = tilesum[x:x+8, y:y+8]
        # Abs diff between tile and one to it's right
        sA = subtile[1:,:][tile_mask[:7,:]] - subtile[:7,:][tile_mask[:7,:]]
        # Abs diff between tile and one below
        sB = subtile[:,1:][tile_mask[:,:7]] - subtile[:,:7][tile_mask[:,:7]]
        score = np.sum(np.abs(sA) + np.abs(sB))

        if score > best_score:
          best_score = score
          best_idx = (x, y) # return in x,y coordinate system

  else:
    #### Old
    # Now try 8x8 subsquares of this 12x12 array and take the 
    # difference between the sum of white and sum of black tiles (abs)
    # and use that as a score.
    # same_color_tile_mask = np.tile(np.eye(2,dtype=bool),[4,4])
    filterA = np.tile(np.eye(2,dtype=np.int64),[4,4])*2-1 # -1 and 1
    # TODO : Add bright boundary around chessboard, since most tend to have a light space there.
    tbuff = 0
    filterA = np.pad(filterA, tbuff, 'constant', constant_values=1)
    # import scipy
    # filterA = scipy.ndimage.filters.gaussian_filter((filterA).astype(np.float64), sigma=10, mode='constant')
    # print(filterA)
    filterB = np.rot90(filterA) # inverse tile order
    # score_array = np.zeros([n_max-8-2*tbuff+1,n_max-8-2*tbuff+1])

    best_score = 0
    best_idx = (-1,-1)
    for i in range(tbuff,n_max-8-tbuff+1):
      for j in range(tbuff,n_max-8-tbuff+1):
        # sA = np.sum(tilesum[i:i+8,j:j+8][same_color_tile_mask])
        # sB = np.sum(tilesum[i:i+8,j:j+8][~same_color_tile_mask])
        # score = np.abs(sA-sB)
        subtile = tilesum[
          i-tbuff:i+8+tbuff, 
          j-tbuff:j+8+tbuff]
        scoreA = np.sum(filterA*subtile)**2
        scoreB = np.sum(filterB*subtile)**2
        score = max(scoreA, scoreB)
        # score_array[i-1,j-1] = score
        if score > best_score:
          best_score = score
          best_idx = (i, j) # return in x,y coordinate system


  tilesum[:best_idx[1],:best_idx[0]] = 127
  # print(best_score)

  # score_array = (score_array * 255) / score_array.max()

  return best_idx, tilesum, best_score

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



def videostream(predict_fn, filepath='carlsen_match.mp4', output_folder_prefix='', SAVE_FRAME=True, MAX_FRAME=None, DO_VISUALS=True):
  print("Loading video %s" % filepath)
  # vidstream = skvideo.io.vread(filepath, num_frames=4000)
  # Load frame-by-frame
  vidstream = skvideo.io.vreader(filepath)
  print("Finished loading")
  # print(vidstream.shape)

  # ffmpeg -i vidstream_frames/ml_frame_%03d.jpg -c:v libx264 -vf "fps=25,format=yuv420p"  test.avi -y
  filename = os.path.basename(filepath)

  if SAVE_FRAME:
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
    if i >= MAX_FRAME:
      print('Reached max frame %d >= %d' % (i, MAX_FRAME))
      break
    print("Frame %d" % i)
    # if (i%5!=0):
    #   continue
    if frame.shape[1] > 640:
      frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_CUBIC)
      # frame = cv2.resize(frame, (960, 720), interpolation = cv2.INTER_CUBIC)
    # frame = cv2.resize(frame, (480, 360), interpolation = cv2.INTER_CUBIC)


    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    a = time.time()
    overlay_frame, warpFrame, chessboard_corners, tilesum = processFrame(frame.copy(), gray, predict_fn)
    t_proc = time.time() - a

    # Add frame counter
    cv2.putText(overlay_frame, 'Frame % 4d (Processed in % 6.1f ms)' % (i, t_proc*1e3), (5,15), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255),0)

    if DO_VISUALS:
      # Display the resulting frame
      cv2.imshow('overlayFrame',overlay_frame)
      if warpFrame is not None:
        cv2.imshow('warpFrame',warpFrame)
      if tilesum is not None:
        # cv2.imshow('tilesum',tilesum/4)
        cv2.imshow('tilemedian',np.clip((tilesum+128), 0, 255).astype(np.uint8))
        # cv2.imshow('tilemedian',(tilesum).astype(np.int64))

    if SAVE_FRAME:
      output_orig_filepath = '%s/frame_%03d.jpg' % (output_folder, i)
      output_filepath = '%s/ml_frame_%03d.jpg' % (output_folder, i)
      output_filepath_warp = '%s/ml_warp_frame_%03d.jpg' % (output_folder, i)
      cv2.imwrite(output_orig_filepath, frame)
      cv2.imwrite(output_filepath, overlay_frame)
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

    if DO_VISUALS:
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  # When everything done, release the capture
  # cap.release()
  if DO_VISUALS:
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
  parser = ArgumentParser()
  parser.add_argument("--model", dest="model", default=None,
                      help="Path to exported model to use.")
  parser.add_argument("video_inputs", nargs='+',
                      help="filepaths to videos to process")
  parser.add_argument("-save_frame",
                      action='store_true', help="Save output frames")
  args = parser.parse_args()
  print("Arguments passed: \n\t%s\n" % args)
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
  # filename = 'john2.mp4' # Slight motion, clean but slow
  # filename = 'swivel.mp4' # Moving around a fancy gold board

  allfiles = ['chess_beer.mp4', 'random1.mp4', 'match2.mp4','output.avi','output.mp4',
    'speedchess1.mp4','wgm_1.mp4','gm_magnus_1.mp4',
    'bro_1.mp4','output2.avi','john1.mp4','john2.mp4','swivel.mp4', 'sam2.mp4']

  # for filename in allfiles:
  # for filename in ['match2.mp4']:
    # fullpath = 'datasets/raw/videos/%s' % filename
  for fullpath in args.video_inputs:
    output_folder_prefix = 'results'
    processFrame.prevBoardpts = None
    processFrame.prevGray = None
    print('\n\n - ON %s\n\n' % fullpath)
    # predict_fn = RunExportedMLOnImage.getModel('ml/model/run97pct/1528942225')
    if (args.model):
      predict_fn = RunExportedMLOnImage.getModel(args.model)
    else:
      predict_fn = RunExportedMLOnImage.getModel()
    videostream(predict_fn, fullpath, output_folder_prefix, args.save_frame, MAX_FRAME=1000, DO_VISUALS=True)