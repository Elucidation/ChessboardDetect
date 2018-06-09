# coding=utf-8
import os
import cv2
import PIL.Image
import skvideo.io
import numpy as np
import Brutesac
from functools import wraps
import time
from scipy.spatial import ConvexHull

@Brutesac.timed
def calculateOnFrame(gray):
  # and return M for chessboard from image
  pts = Brutesac.classifyImage(gray)

  # pts = np.loadtxt('example_pts.txt')
  pts = pts[:,[1,0]] # Switch rows/cols to x/y for plotting on an image

  # Get contours
  contours, hierarchy = getContours(gray, pts)

  # xcorner_map = np.zeros(gray.shape, dtype=np.uint8)
  # for pt in pts:
  #   cv2.circle(xcorner_map, tuple(pt), 5, 1, -1)

  contours, hierarchy = pruneContours(contours, hierarchy, pts)

  return pts, contours

@Brutesac.timed
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


@Brutesac.timed
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

@Brutesac.timed
def getContours(gray, pts, iters=10):
  edges = cv2.Canny(gray, 20, 250)

  # Mask edges to only those in convex hull of points (dilated)
  if len(pts) >= 3:
    xcorner_mask = np.zeros(gray.shape, dtype=np.uint8)
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
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
  pts, contours = calculateOnFrame(gray)

  raw_M, best_quad, best_offset, best_score, best_error_score = contourSacChessboard(pts, contours)
  if raw_M is not None:
    M_homog = Brutesac.refineHomography(pts, raw_M, best_offset)
  else:
    M_homog = None

  # Draw tiles found
  # cv2.drawContours(frame,contours,-1,(0,255,255),2)

  # Draw xcorner points
  for pt in pts:
    cv2.circle(frame, tuple(pt), 3, (0,0,255), -1)


  ideal_grid_pts = np.vstack([np.array([0,0,1,1,0])*8-1, np.array([0,1,1,0,0])*8-1]).T

  if M_homog is not None:
    # Refined via homography of all valid points
    unwarped_ideal_chess_corners_homography = cv2.perspectiveTransform(
        np.expand_dims(ideal_grid_pts.astype(float),0), np.linalg.inv(M_homog))[0,:,:]

    cv2.polylines(frame, 
      [unwarped_ideal_chess_corners_homography.astype(np.int32)], 
      isClosed=True, thickness=4, color=(0,0,255))

  # if best_quad is not None:
    # cv2.polylines(frame, 
    #   [best_quad.astype(np.int32)], 
    #   isClosed=True, thickness=4, color=(255,0,255))

  # Visualize mask used by getContours
  # if len(pts) >= 3:
  #   xcorner_mask = np.zeros(gray.shape, dtype=np.uint8)
  #   hull = ConvexHull(pts)
  #   hull_pts = pts[hull.vertices]
  #   xcorner_mask = cv2.fillConvexPoly(xcorner_mask, hull_pts, 255)
  #   # Dilate mask a bit
  #   element = np.ones([21, 21], np.uint8)
  #   xcorner_mask = cv2.dilate(xcorner_mask, element)

  #   frame = cv2.bitwise_and(frame,frame,mask = xcorner_mask)
  return frame

  # M_homog, pts = Brutesac.calculateOnFrame(gray)

  # if M_homog is not None:
  #   ideal_grid_pts = np.vstack([np.array([0,0,1,1,0])*8-1, np.array([0,1,1,0,0])*8-1]).T
  #   unwarped_ideal_chess_corners_homography = cv2.perspectiveTransform(
  #         np.expand_dims(ideal_grid_pts.astype(float),0), np.linalg.inv(M_homog))[0,:,:]



  #   # for pt in unwarped_ideal_chess_corners_homography:
  #   #   cv2.circle(frame, tuple(pt[::-1]), 3, (0,0,255), -1)
  #   cv2.polylines(frame, [unwarped_ideal_chess_corners_homography.astype(np.int32)], isClosed=True, thickness=3, color=(0,0,255))

  # cv2.putText(frame, 'Frame %d' % i, (5,15), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255),0,cv2.LINE_AA)

def getWarpedChessboard(img, M, tile_px=32):
  # Given a the 4 points of a chessboard, get a warped image of just the chessboard

  # board_pts = np.vstack([
  #   np.array([0,0,1,1])*tile_px,
  #   np.array([0,1,1,0])*tile_px
  #   ]).T
  img_warp = cv2.warpPerspective(img, M, (8*tile_px, 8*tile_px))
  return img_warp



def videostream(filename='carlsen_match.mp4', SAVE_FRAME=True):
  print("Loading video %s" % filename)
  # vidstream = skvideo.io.vread(filename, num_frames=4000)
  # Load frame-by-frame
  vidstream = skvideo.io.vreader(filename)
  print("Finished loading")
  # print(vidstream.shape)

  # ffmpeg -i vidstream_frames/ml_frame_%03d.jpg -c:v libx264 -vf "fps=25,format=yuv420p"  test.avi -y

  output_folder = "%s_vidstream_frames" % (filename[:-4])
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)

  for i, frame in enumerate(vidstream):
    # if i < 2000:
    #   continue
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
    frame = processFrame(frame, gray)
    t_proc = time.time() - a

    # Add frame counter
    cv2.putText(frame, 'Frame % 4d (Processed in % 6.1f ms)' % (i, t_proc*1e3), (5,15), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255),0)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    output_filepath = '%s/ml_frame_%03d.jpg' % (output_folder, i)
    if SAVE_FRAME:
      cv2.imwrite(output_filepath, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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
  # filename = 'carlsen_match.mp4'
  # filename = 'carlsen_match2.mp4'
  # filename = 'output2.avi'
  # filename = 'random1.mp4'
  filename = 'match2.mp4'
  # filename = 'output.avi'
  # filename = 'speedchess1.mp4'
  # filename = 'chess_beer.mp4'
  # filename = 'john1.mp4'
  # filename = 'john2.mp4'
  # filename = 'john3.mp4'
  videostream(filename, False)



