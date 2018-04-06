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

# Set up model
featureA = tf.feature_column.numeric_column("x", shape=[11,11], dtype=tf.uint8)

estimator = tf.estimator.DNNClassifier(
  feature_columns=[featureA],
  hidden_units=[256, 32],
  n_classes=2,
  dropout=0.1,
  model_dir='./xcorner_model_6k',
  )

# Saddle

def getSaddle(gray_img):
    img = gray_img.astype(np.float64)
    gx = cv2.Sobel(img,cv2.CV_64F,1,0)
    gy = cv2.Sobel(img,cv2.CV_64F,0,1)
    gxx = cv2.Sobel(gx,cv2.CV_64F,1,0)
    gyy = cv2.Sobel(gy,cv2.CV_64F,0,1)
    gxy = cv2.Sobel(gx,cv2.CV_64F,0,1)
    
    S = gxx*gyy - gxy**2
    return S


# void nonmaxSupress(Mat &img) {
#     int dilation_size = 5;
#     Mat img_dilate;
#     Mat peaks;
#     Mat notPeaks;
#     Mat nonzeroImg;
#     Mat element = getStructuringElement(MORPH_RECT,
#                                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
#                                        Point( dilation_size, dilation_size ) );
#     // Dilate max value by window size
#     dilate(img, img_dilate, element);
#     // Compare and find where values of dilated vs original image are NOT the same.
#     compare(img, img_dilate, peaks, CMP_EQ);
#     // compare(img, img_dilate, notPeaks, CMP_NE);
#     compare(img, 0, nonzeroImg, CMP_NE);
#     bitwise_and(nonzeroImg, peaks, peaks); // Only keep peaks that are non-zero
    
#     // Remove peaks that are zero
#     // Also set max to 255
#     // compare(img, 0.8, nonzeroImg, CMP_GT);
#     // bitwise_and(nonzeroImg, peaks, peaks); // Only keep peaks that are non-zero
#     // bitwise_not(peaks, notPeaks);
#     // Set all values where not the same to zero. Non-max suppress.
#     bitwise_not(peaks, notPeaks);
#     img.setTo(0, notPeaks);
#     // img.setTo(255, peaks);

# }

def fast_nonmax_sup(img, win=21):
  element = np.ones([win, win], np.uint8)
  img_dilate = cv2.dilate(img, element)
  peaks = cv2.compare(img, img_dilate, cv2.CMP_EQ)
  # nonzeroImg = cv2.compare(img, 0, cv2.CMP_NE)
  # peaks = cv2.bitwise_and(peaks, nonzeroImg)
  peaks[img == 0] = 0
  # notPeaks = cv2.bitwise_not(peaks)

  img[peaks == 0] = 0
  return img



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

def pruneSaddle(s):
    thresh = 128
    score = (s>0).sum()
    while (score > 10000):
        thresh = thresh*2
        s[s<thresh] = 0
        score = (s>0).sum()


def loadImage(filepath):
    img_orig = PIL.Image.open(filepath)
    img_width, img_height = img_orig.size

    # Resize
    aspect_ratio = min(500.0/img_width, 500.0/img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width,new_height), resample=PIL.Image.BILINEAR)
    gray_img = img.convert('L') # grayscale
    img = np.array(img)
    gray_img = np.array(gray_img)
    
    return img, gray_img

def getFinalSaddlePoints(img):
    blur_img = cv2.blur(img, (3,3)) # Blur it
    saddle = getSaddle(blur_img)
    saddle = -saddle
    saddle[saddle<0] = 0
    pruneSaddle(saddle)
    s2 = fast_nonmax_sup(saddle)
    s2[s2<100000]=0
    spts = np.argwhere(s2)
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

def videostream():
  # vidstream = skvideo.io.vread('VID_20170427_003836.mp4')
  # vidstream = skvideo.io.vread('VID_20170109_183657.mp4')
  print("Loading video")
  # vidstream = skvideo.io.vread('output2.avi')
  vidstream = skvideo.io.vread('output.avi')
  print("Finished loading")
  # vidstream = skvideo.io.vread(0)
  print(vidstream.shape)

  output_folder = "vidstream_frames"
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
    # if (i%2!=0):
    #   continue
    
    # frame = cv2.resize(frame, (320,240), interpolation = cv2.INTER_CUBIC)

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    final_predictions, pred_pts = processImage(gray)

    for c,pt in zip(final_predictions, pred_pts):
      if (c == 1):
        # Good
        cv2.circle(frame, tuple(pt[::-1]), 4, (0,255,0), -1)
      else:
        cv2.circle(frame, tuple(pt[::-1]), 2, (0,0,255), -1)

    cv2.putText(frame, 'Frame %d' % i, (5,15), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255),0,cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    output_filepath = '%s/ml_frame_%03d.jpg' % (output_folder, i)
    cv2.imwrite(output_filepath, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  # When everything done, release the capture
  # cap.release()
  cv2.destroyAllWindows()

def processImage(img_gray):
  a = time.time()
  spts = getFinalSaddlePoints(img_gray)
  b = time.time()
  print("getFinalSaddlePoints() took %.2f ms" % ((b-a)*1e3))
  WINSIZE = 5

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
  tiles = np.array(tiles, dtype=np.uint8)

  a = time.time()
  predictions = estimator.predict(input_fn=input_fn_predict(tiles))

  good_pts = []
  bad_pts = []
  final_predictions = []
  for i, prediction in enumerate(predictions):
    c = prediction['probabilities'].argmax()
    pt = pred_pts[i]
    final_predictions.append(c)
  b = time.time()
  print("predict() took %.2f ms" % ((b-a)*1e3))
  return final_predictions, pred_pts




def main():
  filenames = glob.glob('input_bad/*')
  # filenames = glob.glob('input/img_*')
  # filenames.extend(glob.glob('input_yt/*.jpg'))
  filenames = sorted(filenames)
  n = len(filenames)
  n = 1

  WINSIZE = 5

  for i in range(n):
    filename = filenames[i]
    print ("Processing %d/%d : %s" % (i+1,n,filename))
    img, img_gray = loadImage(filename)
    final_predictions, pred_pts = processImage(img_gray)

    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb

    for c,pt in zip(final_predictions, pred_pts):
      if (c == 1):
        # Good
        cv2.circle(rgb_img, tuple(pt[::-1]), 4, (0,255,0), -1)
      else:
        cv2.circle(rgb_img, tuple(pt[::-1]), 3, (0,0,255), -1)

    cv2.imshow('frame',rgb_img)
    if cv2.waitKey() & 0xFF == ord('q'):
      break

  print('Finished')

    




if __name__ == '__main__':
  # main()
  videostream()



