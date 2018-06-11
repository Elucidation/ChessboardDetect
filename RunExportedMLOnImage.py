# coding=utf-8
import PIL.Image
import matplotlib.image as mpimg
import scipy.ndimage
import cv2 # For Sobel etc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time
import sys
import skvideo.io
np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays

from scipy.spatial import Delaunay

from tensorflow.contrib import predictor

# export_dir = 'ml/model/001/1521934334' # old
# export_dir = 'ml/model/002/1528405915' # newer (same dataset, but random image changes)
# export_dir = 'ml/model/003/1528406613' # newer still
# export_dir = 'ml/model/004/1528441286' # win 21x21, 95% accuracy
# export_dir = 'ml/model/005/1528489968' # win 21x21 96% accuracy
export_dir = 'ml/model/006/1528565066' # win 21x21 97% accuracy
predict_fn = predictor.from_saved_model(export_dir, signature_def_key='predict')

# Saddle
def getSaddle(gray_img):
    img = gray_img#.astype(np.float64)
    gx = cv2.Sobel(img,cv2.CV_32F,1,0)
    gy = cv2.Sobel(img,cv2.CV_32F,0,1)
    gxx = cv2.Sobel(gx,cv2.CV_32F,1,0)
    gyy = cv2.Sobel(gy,cv2.CV_32F,0,1)
    gxy = cv2.Sobel(gx,cv2.CV_32F,0,1)
    
    S = -gxx*gyy + gxy**2
    return S

def fast_nonmax_sup(img, win=11):
  element = np.ones([win, win], np.uint8)
  img_dilate = cv2.dilate(img, element)
  peaks = cv2.compare(img, img_dilate, cv2.CMP_EQ)
  # nonzeroImg = cv2.compare(img, 0, cv2.CMP_NE)
  # peaks = cv2.bitwise_and(peaks, nonzeroImg)
  # peaks[img == 0] = 0
  # notPeaks = cv2.bitwise_not(peaks)

  img[peaks == 0] = 0
  # return img



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

def getFinalSaddlePoints(img): # 32ms -> 15ms
    # img = cv2.blur(img, (3,3)) # Blur it (.5ms)
    saddle = getSaddle(img) # 6ms
    # a = time.time()
    # pruneSaddle(saddle, 1024) # (1024 = 8ms)
    # b = time.time()
    # print("getSaddle() took %.2f ms" % ((b-a)*1e3))
    fast_nonmax_sup(saddle) # ~6ms
    saddle[saddle<10000]=0 # Hardcoded ~1ms
    spts = np.argwhere(saddle)
    return spts

# def processSingle(filename='input/img_10.png'):
#   img = loadImage(filename)
#   spts = getFinalSaddlePoints(img)
def input_fn_predict(img_data): # returns x, None
  def ret_func():
    dataset = tf.data.Dataset.from_tensor_slices(
        {
        'x':img_data
        }
      )
    dataset = dataset.batch(25)
    return dataset.make_one_shot_iterator().get_next(), None
  return ret_func

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
#             print(i,j)
            if simplices_mask is not None:
                if simplices_mask[i]:
                    good_neighbors.append(i)
                if simplices_mask[j]:
                    good_neighbors.append(j)
            else:
                good_neighbors.extend([i,j])
    return good_neighbors

def videostream(filename='carlsen_match.mp4', SAVE_FRAME=True):
  # vidstream = skvideo.io.vread('VID_20170427_003836.mp4')
  # vidstream = skvideo.io.vread('VID_20170109_183657.mp4')
  print("Loading video %s" % filename)
  # vidstream = skvideo.io.vread('output2.avi')
  vidstream = skvideo.io.vread(filename)#, num_frames=1000)
  # vidstream = skvideo.io.vread('output.avi')
  print("Finished loading")
  # vidstream = skvideo.io.vread(0)
  print(vidstream.shape)

  # ffmpeg -i vidstream_frames/ml_frame_%03d.jpg -c:v libx264 -vf "fps=25,format=yuv420p"  test.avi -y

  output_folder = "%s_vidstream_frames" % (filename[:-4])
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)

  # cap.set(3,320)
  # cap.set(4,240)

  # while(True):
      # Capture frame-by-frame
      # ret, frame = cap.read()

      # if not ret:
      #   print("No frame, stopping")
      #   break
  for i, frame in enumerate(vidstream):
    # if i < 900:
    #   continue
    print("Frame %d" % i)
    # if (i%5!=0):
    #   continue
    
    # frame = cv2.resize(frame, (320,240), interpolation = cv2.INTER_CUBIC)

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # if i==900:
    #   cv2.imwrite("frame%d.png" % i, frame)

    inlier_pts, outlier_pts, pred_pts, final_predictions, prediction_levels, tri, simplices_mask = processImage(gray)


    for pt in inlier_pts:
      cv2.circle(frame, tuple(pt[::-1]), 3, (0,255,0), -1)

    for pt in outlier_pts:
      cv2.circle(frame, tuple(pt[::-1]), 1, (0,0,255), -1)

    # Draw triangle mesh
    if tri is not None:
      cv2.polylines(frame,
        np.flip(inlier_pts[tri.simplices].astype(np.int32), axis=2),
        isClosed=True, color=(255,0,0))
      cv2.polylines(frame,
        np.flip(inlier_pts[tri.simplices[simplices_mask]].astype(np.int32), axis=2),
        isClosed=True, color=(0,255,0), thickness=3)

    cv2.putText(frame, 'Frame %d' % i, (5,15), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255),0,cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    output_filepath = '%s/ml_frame_%03d.jpg' % (output_folder, i)
    if SAVE_FRAME:
      cv2.imwrite(output_filepath, frame)

    # if i==900:
    #   cv2.imwrite("frame%d_overlay.png" % i, frame)
    #   print(np.array(pred_pts)[np.array(prediction_levels)>0.8,:])
    #   break;

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  # When everything done, release the capture
  # cap.release()
  cv2.destroyAllWindows()

def calculateOutliers(pts, threshold_mult = 2.5):
  N = len(pts)
  std = np.std(pts, axis=0)
  ctr = np.mean(pts, axis=0)
  return (np.any(np.abs(pts-ctr) > threshold_mult * std, axis=1))

def processImage(img_gray):
  print("Processing.")
  a_start = time.time()
  spts = getFinalSaddlePoints(img_gray)
  b = time.time()
  t_saddle = (b-a_start)
  WINSIZE = 10

  tiles = []
  pred_pts = []
  for pt in spts:
    # Build tiles
    if (np.any(pt <= WINSIZE) or np.any(pt >= np.array(img_gray.shape[:2]) - WINSIZE)):
      continue
    else:
      tile = img_gray[pt[0]-WINSIZE:pt[0]+WINSIZE+1, pt[1]-WINSIZE:pt[1]+WINSIZE+1]
      tiles.append(tile)
      pred_pts.append(pt)

  if tiles == []:
    return [None]*7

  tiles = np.array(tiles, dtype=np.uint8)

  a = time.time()
  predictions = predict_fn(
    {"x": tiles})

  # print(predictions)
  # raise('Error')

  good_pts = []
  bad_pts = []
  final_predictions = []
  prediction_levels = []
  # for i, prediction in enumerate(predictions['probabilities']):
    # c = prediction.argmax()
  for i, prediction in enumerate(predictions['class_ids']):
    c = prediction[0]
    pt = pred_pts[i]
    final_predictions.append(c)
    # prediction_levels.append(prediction[1])
    prediction_levels.append(c)
  b = time.time()
  t_pred = b-a

  ml_pts = np.array(pred_pts)[np.array(prediction_levels)>0.5,:]
  bad_pts_mask = calculateOutliers(ml_pts)
  # Inliers
  inlier_pts = ml_pts[~bad_pts_mask,:]
  outlier_pts = ml_pts[bad_pts_mask,:]

  if (len(inlier_pts) >= 4):
    tri = Delaunay(inlier_pts)
    dists, good_simplices_mask = removeOutlierSimplices(tri)
    good_neighbors = findQuadSimplices(tri, dists, good_simplices_mask)
    simplices_mask = good_simplices_mask
    # simplices_mask = np.zeros(tri.nsimplex, dtype=bool)
    # simplices_mask[good_neighbors] = True
  else:
    tri = None
    simplices_mask = None

  b_end = time.time()
  t_total = b_end-a_start
  print(" - Total %.2f ms, Saddle took %.2f ms (%d pts), Predict took %.2f ms" % (t_total*1e3, t_saddle*1e3, len(spts), t_pred*1e3)) # ~2-3ms

  return inlier_pts, outlier_pts, pred_pts, final_predictions, prediction_levels, tri, simplices_mask




def main(SAVE_FRAME=False):
  # filenames = glob.glob('input_bad/*')
  filenames = glob.glob('input/img_*')
  filenames.extend(glob.glob('input_yt/*.jpg'))
  # filenames = (glob.glob('input_yt/*.jpg'))
  filenames = sorted(filenames)
  n = len(filenames)
  # n = 5

  WINSIZE = 5

  output_folder = 'ml_images'
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)

  for i in range(n):
    filename = filenames[i]
    print ("Processing %d/%d : %s" % (i+1,n,filename))
    img, img_gray = loadImage(filename)
    inlier_pts, outlier_pts, pred_pts, final_predictions, prediction_levels, tri, simplices_mask = processImage(img_gray)

    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    for pt in inlier_pts:
      # Good
      cv2.circle(rgb_img, tuple(pt[::-1]), 4, (0,255,0), -1)
    for pt in outlier_pts:
      cv2.circle(rgb_img, tuple(pt[::-1]), 3, (0,0,255), -1)

    # Draw triangle mesh
    if tri is not None:
      cv2.polylines(rgb_img,
        np.flip(inlier_pts[tri.simplices].astype(np.int32), axis=2),
        isClosed=True, color=(255,0,0))
      cv2.polylines(rgb_img,
        np.flip(inlier_pts[tri.simplices[simplices_mask]].astype(np.int32), axis=2),
        isClosed=True, color=(0,255,0), thickness=3)

    cv2.imshow('frame',rgb_img)

    output_filepath = '%s/%s_ml.png' % (output_folder,filename[:-3])
    if SAVE_FRAME:
      if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath))
      print("Saving %s" % output_filepath)
      cv2.imwrite(output_filepath, rgb_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  print('Finished')



if __name__ == '__main__':
  # main(True)
  # filename = 'carlsen_match.mp4'
  # filename = 'carlsen_match2.mp4'
  filename = 'output.avi'
  # filename = 'output2.avi'
  # filename = 'random1.mp4'
  # filename = 'speedchess1.mp4'
  # filename = 'match1.mp4'
  # filename = 'match2.mp4'
  videostream(filename, True)



