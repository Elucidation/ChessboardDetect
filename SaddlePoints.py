# coding=utf-8
# Saddle point calculator.
import cv2 # For Sobel etc
import numpy as np


# Saddle
def getSaddle(gray_img):
    img = gray_img
    gx = cv2.Sobel(img,cv2.CV_32F,1,0)
    gy = cv2.Sobel(img,cv2.CV_32F,0,1)
    gxx = cv2.Sobel(gx,cv2.CV_32F,1,0)
    gyy = cv2.Sobel(gy,cv2.CV_32F,0,1)
    gxy = cv2.Sobel(gx,cv2.CV_32F,0,1)
    
    # Inverse everything so positive equals more likely.
    S = -gxx*gyy + gxy**2

    # Calculate subpixel offsets
    denom = (gxx*gyy - gxy*gxy)
    sub_s = np.divide(gy*gxy - gx*gyy, denom, out=np.zeros_like(denom), where=denom!=0)
    sub_t = np.divide(gx*gxy - gy*gxx, denom, out=np.zeros_like(denom), where=denom!=0)
    return S, sub_s, sub_t, gx, gy

def fast_nonmax_sup(img, win=11):
  element = np.ones([win, win], np.uint8)
  img_dilate = cv2.dilate(img, element)
  peaks = cv2.compare(img, img_dilate, cv2.CMP_EQ)
  img[peaks == 0] = 0

def getFinalSaddlePoints(img, WINSIZE=10): # 32ms -> 15ms
  # Get all saddle points that are not closer than WINSIZE to the boundaries.
  
  img = cv2.blur(img, (3,3)) # Blur it (.5ms)
  saddle, sub_s, sub_t, gx, gy = getSaddle(img) # 6ms
  fast_nonmax_sup(saddle) # ~6ms

  # Strip off low points
  saddle[saddle<10000]=0 # Hardcoded ~1ms
  sub_idxs = np.nonzero(saddle)
  spts = np.argwhere(saddle).astype(np.float64)[:,[1,0]] # Return in x,y order instead or row-col

  # Add on sub-pixel offsets
  subpixel_offset = np.array([sub_s[sub_idxs], sub_t[sub_idxs]]).transpose()
  spts = spts + subpixel_offset
  
  # Remove those points near win_size edges
  spts = clipBoundingPoints(spts, img.shape, WINSIZE)

  return spts, gx, gy # returns in x,y column order

def clipBoundingPoints(pts, img_shape, WINSIZE=10): # ~100us
  # Points are given in x,y coords, not r,c of the image shape
  a = ~np.any(np.logical_or(pts <= WINSIZE, pts[:,[1,0]] >= np.array(img_shape)-WINSIZE-1), axis=1)
  return pts[a,:]