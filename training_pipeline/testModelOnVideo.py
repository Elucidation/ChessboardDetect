# coding=utf-8
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
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

@Brutesac.timed
def classifyFrame(gray, predict_fn, WINSIZE=10):
  # All saddle points
  spts = RunExportedMLOnImage.getFinalSaddlePoints(gray)

  # Saddle points classified as Chessboard X-corners
  probabilities = Brutesac.predictOnImage(spts, gray, predict_fn, WINSIZE=WINSIZE)

  return spts, probabilities

@Brutesac.timed
def processFrame(frame, gray, predict_fn, probability_threshold=0.9,WINSIZE=10):
  overlay_frame = frame.copy()
  # Overlay good and bad points onto the frame
  spts, probabilities = classifyFrame(gray, predict_fn, WINSIZE=WINSIZE)

  # 10ms for the rest of this

  # Threshold over 50% probability as xpts
  xpts = spts[probabilities > probability_threshold,:]
  not_xpts = spts[probabilities <= probability_threshold,:]

  # Draw xcorner points
  for pt in np.round(xpts).astype(np.int64):
    cv2.rectangle(overlay_frame, tuple(pt-2),tuple(pt+2), (0,255,0), -1)

  # Draw rejects
  for pt in np.round(not_xpts).astype(np.int64):
    cv2.rectangle(overlay_frame, tuple(pt-0),tuple(pt+0), (0,0,255), -1)

  return overlay_frame, spts, probabilities


def videostream(predict_fn, filepath='carlsen_match.mp4', 
  output_folder_prefix='', SAVE_FRAME=True, MAX_FRAME=None, 
  DO_VISUALS=True, EVERY_N_FRAMES=1):
  print("Loading video %s" % filepath)
  
  # Load frame-by-frame
  vidstream = skvideo.io.vreader(filepath)
  filename = os.path.basename(filepath)

  output_folder = "%s/%s_vidstream_frames" % (output_folder_prefix, filename[:-4])
  if SAVE_FRAME:
    if not os.path.exists(output_folder):
      os.mkdir(output_folder)

  #   # Set up pts.txt, first line is the video filename
  #   # Following lines is the frame number and the flattened M matrix for the chessboard
  #   output_filepath_pts = '%s/xpts.txt' % (output_folder)
  #   with open(output_filepath_pts, 'w') as f:
  #     f.write('%s\n' % filepath)

  for i, frame in enumerate(vidstream):
    if i >= MAX_FRAME:
      print('Reached max frame %d >= %d' % (i, MAX_FRAME))
      break
    print("Frame %d" % i)
    if (i%EVERY_N_FRAMES!=0):
      continue
    
    # # Resize to 960x720
    frame = cv2.resize(frame, (480, 360), interpolation = cv2.INTER_CUBIC)

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    a = time.time()
    overlay_frame, spts, probabilities = processFrame(frame, gray, predict_fn, WINSIZE=7)
    t_proc = time.time() - a

    # Add frame counter
    cv2.putText(overlay_frame, 'Frame % 4d (Processed in % 6.1f ms)' % (i, t_proc*1e3), (5,15), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255),0)

    if DO_VISUALS:
      # Display the resulting frame
      cv2.imshow('overlayFrame',overlay_frame)
    

    output_orig_filepath = '%s/frame_%03d.jpg' % (output_folder, i)
    output_filepath = '%s/ml_frame_%03d.jpg' % (output_folder, i)

    if SAVE_FRAME:
      cv2.imwrite(output_orig_filepath, frame)
      cv2.imwrite(output_filepath, overlay_frame)

      # Append line of frame index and chessboard_corners matrix
      # if chessboard_corners is not None:
      #   with open(output_filepath_pts, 'a') as f:
      #     chessboard_corners_str = ','.join(map(str,spts.flatten()))
      #     # M_str = M.tostring() # binary
      #     f.write(u'%d,%s\n' % (i, chessboard_corners_str))

    if DO_VISUALS:
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  # When everything done, release the capture
  if DO_VISUALS:
    cv2.destroyAllWindows()


def main():
  filename = 'chess_out1.png'

  print ("Processing %s" % (filename))
  img = PIL.Image.open(filename).resize([600,400])
  rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  gray = np.array(img.convert('L'))

  ###
  overlay_frame, spts, probabilities  = processFrame(rgb, gray)
  ###

  cv2.imshow('frame',overlay_frame)
  cv2.waitKey()

  print('Finished')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--model", dest="model", default='',
                      help="Path to exported model to use.")
  parser.add_argument("video_inputs", nargs='+',
                      help="filepaths to videos to process")


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
  for fullpath in args.video_inputs:
    # fullpath = 'datasets/raw/videos/%s' % filename
    output_folder_prefix = '../results'
    print('\n\n - ON %s\n\n' % fullpath)
    # predict_fn = RunExportedMLOnImage.getModel()
    # predict_fn = RunExportedMLOnImage.getModel('ml/model/run97pct/1528942225')
    predict_fn = RunExportedMLOnImage.getModel(args.model)
    videostream(predict_fn, fullpath, output_folder_prefix, 
      SAVE_FRAME=False, MAX_FRAME=1000, DO_VISUALS=True)