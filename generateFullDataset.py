# Given a list of pts text files, build a complete dataset from it.
import glob
import os
import PIL.Image
import cv2
import numpy as np
from time import time
from argparse import ArgumentParser
from scipy.spatial import cKDTree
import tensorflow as tf
import SaddlePoints
import errno

def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if os.path.isdir(path):
      pass
    else:
      raise


# Given chessboard corners, get all 7x7 = 49 internal x-corner positions.
def getXcorners(corners):
  # Get Xcorners for image
  ideal_corners = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
  M = cv2.getPerspectiveTransform(ideal_corners, corners) # From ideal to real.

  # 7x7 internal grid of 49 x-corners/
  xx,yy = np.meshgrid(np.arange(7, dtype=np.float32), np.arange(7, dtype=np.float32))
  all_ideal_grid_pts = np.vstack([xx.flatten(), yy.flatten()]).T
  all_ideal_grid_pts = (all_ideal_grid_pts + 1) / 8.0

  chess_xcorners = cv2.perspectiveTransform(np.expand_dims(all_ideal_grid_pts,0), M)[0,:,:]
  return chess_xcorners


def getPointsNearPoints(ptsA, ptsB, MIN_DIST_PX=3):
  # Returns a mask for points in A that are close by MIN_DIST_PX to points in B
  min_dists, min_dist_idx = cKDTree(ptsB).query(ptsA, 1)
  mask = min_dists < MIN_DIST_PX
  return mask

# Load image from path
def loadImage(img_filepath):
  print ("Processing %s" % (img_filepath))
  
  img = PIL.Image.open(img_filepath)
  if (img.size[0] > 640):
    img = img.resize((640, 480), PIL.Image.BICUBIC)
  gray = np.array(img.convert('L'))
  rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  return rgb, gray

def getTiles(pts, img_gray, WINSIZE=10):
  # NOTE : Assumes no point is within WINSIZE of an edge!
  # Points Nx2, columns should be x and y, not r and c.
  # WINSIZE = the number of pixels out from the point that a tile should be.

  # Build tiles of size Nx(2*WINSIZE+1)x(2*WINSIZE+1)
  img_shape = np.array([img_gray.shape[1], img_gray.shape[0]])
  tiles = np.zeros([len(pts), WINSIZE*2+1, WINSIZE*2+1], dtype=img_gray.dtype)
  for i, pt in enumerate(np.round(pts).astype(np.int64)):
    tiles[i,:,:] = img_gray[pt[1]-WINSIZE:pt[1]+WINSIZE+1,
                            pt[0]-WINSIZE:pt[0]+WINSIZE+1]
  return tiles

def getTilesColor(pts, img, WINSIZE=10):
  # NOTE : Assumes no point is within WINSIZE of an edge!
  # Points Nx2, columns should be x and y, not r and c.
  # WINSIZE = the number of pixels out from the point that a tile should be.

  # Build tiles of size Nx(2*WINSIZE+1)x(2*WINSIZE+1)
  img_shape = np.array([img.shape[1], img.shape[0]])
  tiles = np.zeros([len(pts), WINSIZE*2+1, WINSIZE*2+1, 3], dtype=img.dtype)
  for i, pt in enumerate(np.round(pts).astype(np.int64)):
    tiles[i,:,:,:] = img[pt[1]-WINSIZE:pt[1]+WINSIZE+1,
                            pt[0]-WINSIZE:pt[0]+WINSIZE+1, :]
  return tiles

# View image with chessboard lines overlaid.
def addOverlay(idx, img, corners, good_xcorners, bad_pts):
  for pt in np.round(bad_pts).astype(np.int64):
    cv2.rectangle(img, tuple(pt-2),tuple(pt+2), (0,0,255), -1)

  for pt in np.round(good_xcorners).astype(np.int64):
    cv2.rectangle(img, tuple(pt-2),tuple(pt+2), (0,255,0), -1)


  cv2.polylines(img, 
      [np.round(corners).astype(np.int32)], 
      isClosed=True, thickness=2, color=(255,0,255))

  cv2.putText(img, 
    'Frame % 4d' % (idx),
    (5,15), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255),0)

def visualizeTiles(tiles):
  # Assumes no more than 49 tiles, only plots the first 49
  N = len(tiles)
  # assert N <= 49
  assert tiles.shape[1] == tiles.shape[2] # square tiles
  side = tiles.shape[1]
  cols = 7#int(np.ceil(np.sqrt(N)))
  rows = 7#int(np.ceil(N/(cols)))+1
  tile_img = np.zeros([rows*side, cols*side, 3], dtype=tiles.dtype)
  for i in range(min(N,49)):
    r, c = side*(int(i/cols)), side*(i%cols)
    tile_img[r:r+side, c:c+side,:] = tiles[i,:,:,:]
  return tile_img

# Converting the values into features
# _int64 is used for numeric values
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(args):
  for pointfile in args.pointfiles:
    with open(pointfile, 'r') as f:
      lines = f.readlines()
    video_filepath = lines[0]
    images_path = os.path.dirname(pointfile)

    # Writing to TFrecord
    video_filename = os.path.basename(video_filepath)[:-5]
    folder_path = "%s/winsize_%s_color" % (args.tfrecords_path, args.winsize)
    mkdir_p(folder_path)

    tfrecord_path = "%s/%s_ws%d.tfrecords" % (folder_path, video_filename, args.winsize)
    with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
      for line in lines[1:]:
        tA = time()
        parts = line.split(',')
        idx = int(parts[0])

        # if (idx < 260):
        #   continue

        corners = np.array(parts[1:], dtype=np.float32).reshape([4,2])
        xcorners = getXcorners(corners)

        filename = "%s/frame_%03d.jpg" % (images_path, idx)
        img, gray = loadImage(filename)

        # Saddle points
        spts, gx, gy = SaddlePoints.getFinalSaddlePoints(gray, WINSIZE=args.winsize)

        good_spt_mask = getPointsNearPoints(spts, xcorners)
        good_xcorners = spts[good_spt_mask]
        bad_spts = spts[~good_spt_mask]

        # Only keep the same # of bad points as good
        # Shuffle bad points so we get a good smattering.
        N = len(good_xcorners)
        np.random.shuffle(bad_spts)
        bad_spts = bad_spts[:N]

        # good_xcorners, bad_xcorners, bad_spts, spts, keep_mask = getXcornersNearSaddlePts(gray, xcorners)

        tiles = getTilesColor(good_xcorners, img, WINSIZE=args.winsize)
        bad_tiles = getTilesColor(bad_spts, img, WINSIZE=args.winsize)

        # Write tiles to tf-records
        for tile in tiles:
          feature = { 'label': _int64_feature(1),
                      'image': _bytes_feature(tf.compat.as_bytes(tile.tostring())) }
          example = tf.train.Example(features=tf.train.Features(feature=feature))
          writer.write(example.SerializeToString())

        for tile in bad_tiles:
          feature = { 'label': _int64_feature(0),
                      'image': _bytes_feature(tf.compat.as_bytes(tile.tostring())) }
          example = tf.train.Example(features=tf.train.Features(feature=feature))
          writer.write(example.SerializeToString())


        if args.viztiles:
          tile_img = visualizeTiles(tiles)
          bad_tile_img = visualizeTiles(bad_tiles)

        print('\t Took %.1f ms.' % ((time() - tA)*1000))

        if args.vizoverlay:
          overlay_img = img.copy()
          addOverlay(idx, overlay_img, corners, good_xcorners, bad_spts)

          cv2.imshow('frame',overlay_img)
        
        if args.viztiles:
          cv2.imshow('tiles', tile_img)
          cv2.imshow('bad_tiles', bad_tile_img)

        if (args.vizoverlay or args.viztiles):
          if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("pointfiles", nargs='+',
                      help="All pts.txt points files containing filename and chessboard coordinates.")
  parser.add_argument("-savetf",
                      action='store_true', help="Whether to save tfrecords")
  parser.add_argument("-viztiles",
                      action='store_true', help="Whether to visualize tiles or not")
  parser.add_argument("-vizoverlay",
                      action='store_true', help="Whether to visualize overlay")
  parser.add_argument("--tfrecords_path", default='datasets/tfrecords',
                      help="Folder to store tfrecord output")
  parser.add_argument("-ws", "--winsize", dest="winsize", default=10, type=int,
                      help="Half window size (full kernel = 2*winsize + 1)")
  args = parser.parse_args()
  print(args)
  main(args)