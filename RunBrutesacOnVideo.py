# coding=utf-8
import os
import glob
import PIL.Image
import cv2
import skvideo.io
import numpy as np
import Brutesac

def videostream(filename='carlsen_match.mp4', SAVE_FRAME=True):
  print("Loading video %s" % filename)
  vidstream = skvideo.io.vread(filename)#, num_frames=1000)
  print("Finished loading")
  print(vidstream.shape)

  # ffmpeg -i vidstream_frames/ml_frame_%03d.jpg -c:v libx264 -vf "fps=25,format=yuv420p"  test.avi -y

  output_folder = "%s_vidstream_frames" % (filename[:-4])
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)

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

    M_homog, pts = Brutesac.processFrame(gray)

    if M_homog is not None:
      ideal_grid_pts = np.vstack([np.array([0,0,1,1,0])*8-1, np.array([0,1,1,0,0])*8-1]).T
      unwarped_ideal_chess_corners_homography = cv2.perspectiveTransform(
            np.expand_dims(ideal_grid_pts.astype(float),0), np.linalg.inv(M_homog))[0,:,:]


      for pt in pts:
        cv2.circle(frame, tuple(pt), 3, (0,255,0), -1)

      # for pt in unwarped_ideal_chess_corners_homography:
      #   cv2.circle(frame, tuple(pt[::-1]), 3, (0,0,255), -1)
      cv2.polylines(frame, [unwarped_ideal_chess_corners_homography.astype(np.int32)], isClosed=True, thickness=3, color=(0,0,255))

    cv2.putText(frame, 'Frame %d' % i, (5,15), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255),0,cv2.LINE_AA)

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
  filename = 'input/img_01.jpg'

  print ("Processing %s" % (filename))
  gray_img = PIL.Image.open(filename).convert('L').resize([600,400])
  gray = np.array(gray_img)



  cv2.imshow('frame',gray)
  cv2.waitKey()

  print('Finished')

if __name__ == '__main__':
  main()
  # filename = 'carlsen_match.mp4'
  # filename = 'carlsen_match2.mp4'
  # filename = 'output.avi'
  # filename = 'output2.avi'
  # filename = 'random1.mp4'
  # filename = 'speedchess1.mp4'
  # filename = 'match1.mp4'
  # filename = 'match2.mp4'
  # videostream(filename, True)



