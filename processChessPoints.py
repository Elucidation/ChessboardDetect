# coding=utf-8
import PIL.Image
import matplotlib.image as mpimg
import scipy.ndimage
import cv2 # For Sobel etc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays

def process(cpts, pts):
  # Given chess points (cpts) and all saddle points (pts)
  # Find the closest saddle point for each chess point and return the index
  closest_idxs = np.zeros(cpts.shape[0], dtype=np.int)
  best_dists = np.zeros(cpts.shape[0])
  for i,cpt in enumerate(cpts):
    dists = np.sum((pts - cpt)**2, axis=1)
    closest_idxs[i] = np.argmin(dists)
    best_dists[i] = dists[closest_idxs[i]]
  return closest_idxs, best_dists

def loadImage(filepath):
    img_orig = PIL.Image.open(filepath)
    img_width, img_height = img_orig.size

    # Resize
    aspect_ratio = min(500.0/img_width, 500.0/img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width,new_height), resample=PIL.Image.BILINEAR)
    # img = img.convert('L') # grayscale
    img = np.array(img)
    
    return img

def makeProcessedImage(filename, chess_pts, all_pts, closest_idxs, best_dists):
  filename_img = 'input/%s.png' % filename[filename.find('/')+1:filename.find('.')]
  if not os.path.exists(filename_img):
    filename_img = 'input/%s.jpg' % filename[filename.find('/')+1:filename.find('.')]
  if not os.path.exists(filename_img):
    filename_img = 'input_yt/%s.jpg' % filename[filename.find('/')+1:filename.find('.')]
  if not os.path.exists(filename_img):
    filename_img = 'input_yt/%s.png' % filename[filename.find('/')+1:filename.find('.')]
  img = loadImage(filename_img)

  # Only show good updated saddle points
  for i, pt in enumerate(chess_pts):
    if (best_dists[i] <= 2):
      spt = tuple(all_pts[closest_idxs[i]].astype(np.int))
      cv2.circle(img, spt, 2, (0,255,0), -1)
  for i, pt in enumerate(all_pts):
    if (i not in closest_idxs):
      cv2.circle(img, tuple(pt.astype(np.int)), 1, (255,0,0), -1)
  
  # Visualize all
  # for pt in all_pts:
  #   cv2.circle(img, tuple(pt.astype(np.int)), 1, (255,0,0), -1)
  # for i, pt in enumerate(chess_pts):
  #   if (best_dists[i] > 2):
  #     cv2.circle(img, tuple(pt.astype(np.int)), 2, (0,100,0), -1)
  #   else:
  #     cv2.circle(img, tuple(pt.astype(np.int)), 2, (0,255,0), -1)

  # for i in range(len(chess_pts)):
  #   pt_a = tuple(chess_pts[i].astype(np.int))
  #   pt_b = tuple(all_pts[closest_idxs[i]].astype(np.int))
  #   if (best_dists[i] > 2):
  #     cv2.putText(img,'%.1f' % best_dists[i] ,pt_a, font, 0.5,(255,255,255),0,cv2.LINE_AA)
  #     cv2.line(img, pt_a, pt_b, (255,0,255), 1)
  #   else:
  #     cv2.line(img, pt_a, pt_b, (0,0,255), 1)

  im = PIL.Image.fromarray(img).convert('RGB')
  processed_img_filename = filename[:filename.find('.')]
  im.save('%s_proc.png' % processed_img_filename)

def main():
  font = cv2.FONT_HERSHEY_PLAIN
  # all_pts
  filenames_chesspts = glob.glob('positions/*[!_all].txt')
  filenames_chesspts = sorted(filenames_chesspts)
  n_all = len(filenames_chesspts)

  to_skip = [5,7,16,27,28,36,37,38]

  all_good_pts = {}
  all_bad_pts = {}

  for i in range(n_all):
    filename = filenames_chesspts[i]
    filename_short = filename[filename.find('/')+1:filename.find('.')]
    if any('img_%02d' % skip_name in filename for skip_name in to_skip):
      print('Skipping %s' % filename)
      continue
    print ("Processing %d/%d : %s" % (i+1,n_all,filename))
    filename_allpts = filename[:filename.find('.')] + '_all.txt'

    # Load chess points
    chess_pts = np.loadtxt(filename)
    
    # Load all saddle points
    all_pts = np.loadtxt(filename_allpts)
    # all_pts = all_pts[:,[1,0]]
    closest_idxs, best_dists = process(chess_pts, all_pts)
    # print(best_dists)

    makeProcessedImage(filename, chess_pts, all_pts, closest_idxs, best_dists)


    # good_pts = chess_pts[best_dists <= 2, :]
    good_pts = all_pts[closest_idxs[best_dists <= 2],:]
    bad_pts = all_pts.copy()
    bad_pts = np.delete(bad_pts, closest_idxs, 0)
    # print(len(good_pts), len(bad_pts))
    all_good_pts[filename_short] = good_pts.astype(int)
    all_bad_pts[filename_short] = bad_pts.astype(int)
    print(len(good_pts), len(bad_pts))


  # save all points to a file
  with open('pt_dataset2.txt', 'w') as f:
    for filename in sorted(all_good_pts.keys()):
      s0 = ' '.join(all_good_pts[filename][:,0].astype(str))
      s1 = ' '.join(all_good_pts[filename][:,1].astype(str))
      s2 = ' '.join(all_bad_pts[filename][:,0].astype(str))
      s3 = ' '.join(all_bad_pts[filename][:,1].astype(str))
      f.write('\n'.join([filename, s0, s1, s2, s3])+'\n')        

  num_good = 0
  num_bad = 0
  for filename in all_good_pts:
    num_good += all_good_pts[filename].shape[0]
    num_bad += all_bad_pts[filename].shape[0]
  print('Collected %d true and %d false positives' % (num_good, num_bad))
if __name__ == '__main__':
  main()



