import numpy as np

def scaleImageIfNeeded(img, max_width=1000, max_height=800):
  """Scale image down to max_width / max_height keeping aspect ratio if needed. Do nothing otherwise."""
  # Input and Output is a PIL Image
  img_width, img_height = img.size
  # print("Image size %dx%d" % (img_width, img_height))
  aspect_ratio = min(max_width/img_width, max_height/img_height)
  if aspect_ratio < 1.0:
    new_width, new_height = ((np.array(img.size) * aspect_ratio)).astype(int)
    # print(" Resizing to %dx%d" % (new_width, new_height))
    return img.resize((new_width,new_height))
  return img

def getSegmentAngle(line):
  x1,y1,x2,y2 = line
  return np.math.atan2(y2-y1, x2-x1)

def getLineGradients(line, gradient_x, gradient_y, gradient_mag, sampling_rate=0.5,
  normal_magnitude_threshold=0.9, normal_strong_gradient_ratio_threshold=0.45, pos_neg_ratio_threshold=0.2):
  """Calculate normal gradient values along line given x/y gradients and a line segment."""
  
  # 1 - Get gradient values
  line = np.array(line)
  ptA = line[:2]
  ptB = line[2:]
  
  # unit vector in direction of line
  line_length = np.linalg.norm(ptB - ptA)
  line_direction = (ptB - ptA) / line_length
  
  # Convert to normal
  line_normal = np.array([-line_direction[1], line_direction[0]]) # -y, x for normal in one direction

  # Get points along line, choosing number of points giving a sampling rate in pixels per points (1-1 is good)
  num_pts_on_line = np.ceil(np.sqrt(np.sum((ptB - ptA)**2)) / sampling_rate)
  guessx = np.linspace(ptA[1],ptB[1],num_pts_on_line)
  guessy = np.linspace(ptA[0],ptB[0],num_pts_on_line)
  
  line_indices = np.floor(np.vstack((guessx, guessy)).T).astype(int)
  gradients = np.vstack(
          [gradient_x[line_indices[:,0], line_indices[:,1]],
           gradient_y[line_indices[:,0], line_indices[:,1]]])
  gradient_mags = gradient_mag[line_indices[:,0], line_indices[:,1]]
  
  # Calculate average strength of gradient along line as a score
  gradient_on_normal = line_normal.dot(gradients)
  avg_normal_gradient = np.abs(gradient_on_normal).mean()
      
  # Magnitude of gradient along normal, normalized by total gradient magnitude at that point
  # ex. 1.0 means strong + gradient perfectly normal to line
  with np.errstate(divide='ignore', invalid='ignore'):
    # some gradient magnitudes are zero, set normal gradients to zero for those
    normal_gradients = gradient_on_normal / gradient_mags
    # Ignore weak gradients
    normal_gradients[(gradient_mags < 10)] = 0

  # Ratio of how often the absolute of the gradient is greater than the threshold
  # chessboard lines should have extremely high ~90% ratios
  normal_strong_gradient_ratio = np.sum(abs(normal_gradients) > normal_magnitude_threshold) / float(num_pts_on_line)

  # Ratio of mag near top vs mag near bot
  pos_edge_ratio = np.sum(normal_gradients > normal_magnitude_threshold) / float(num_pts_on_line)
  neg_edge_ratio = np.sum(-normal_gradients > normal_magnitude_threshold) / float(num_pts_on_line)

  # Calculate aspect ratio of positive to negative values, but only if they're both reasonably strong
  if (pos_edge_ratio < pos_neg_ratio_threshold or neg_edge_ratio < pos_neg_ratio_threshold):
    edge_ratio = 0
  else:
    if pos_edge_ratio > neg_edge_ratio:
      edge_ratio = neg_edge_ratio / pos_edge_ratio
    else:
      edge_ratio = pos_edge_ratio / neg_edge_ratio
  
  # if normal_strong_gradient_ratio > normal_strong_gradient_ratio_threshold and edge_ratio != 0:
  # print("%.2f / %.2f = %.2f %s | %.2f" % (pos_edge_ratio, neg_edge_ratio, edge_ratio, edge_ratio < 0.75, normal_strong_gradient_ratio))
  

  # Calculate fft, since sampling rate is static, we can just use indices as a comparison method
  fft_result = np.abs(np.fft.rfft(normal_gradients).real)
  strongest_freq = np.argmax(fft_result)

  is_good = True

  # Sanity check normal gradients span positive and negative range
  if normal_gradients.min() > -normal_magnitude_threshold or normal_gradients.max() < normal_magnitude_threshold:
    is_good = False
  # Sanity check most of the normal gradients are maximized
  elif normal_strong_gradient_ratio < normal_strong_gradient_ratio_threshold:
    is_good = False
  # Check that ratio of positive to negative is somewhere near 50/50, 1.0 means perfect 50/50, 0.5 gives a lot of leeway
  elif edge_ratio < 0.5:
    is_good = False
  # Check that there is a low frequency signal in normal gradient
  elif strongest_freq < 2 or strongest_freq > 20:
    is_good = False
  
  # Recover potentially good lines  
  if edge_ratio > 0.9 and normal_strong_gradient_ratio > normal_strong_gradient_ratio_threshold:
    is_good = True

  # if is_good:
  #   print("%.2f : %.2f / %.2f = %.2f | %.2f | %d" % (
  #     avg_normal_gradient, pos_edge_ratio, neg_edge_ratio, edge_ratio, 
  #     normal_strong_gradient_ratio, strongest_freq))
  
  return is_good, strongest_freq, normal_gradients, fft_result, edge_ratio, avg_normal_gradient

def angleClose(a, b, angle_threshold=10*np.pi/180):
  d = np.abs(a - b)
  # Handle angles that are ~360 or ~180 degrees apart
  return d < angle_threshold or np.abs(np.pi-d) < angle_threshold or np.abs(2*np.pi-d) < angle_threshold

def segmentAngles(angles, good_mask, angle_threshold=15*np.pi/180):
  # Partition lines based on similar angles int segments/groups
  good = np.zeros(len(angles),dtype=bool)
  segment_mask = np.zeros(angles.shape, dtype=int)
  
  segment_idx = 1
  for i in range(angles.size):
    # Skip if not a good line or line already grouped
    if not good_mask[i] or segment_mask[i] != 0:
      continue
    
    # Create new group
    segment_mask[i] = segment_idx
    for j in range(i+1, angles.size):
      # If good line, not yet grouped, and is close in angle, add to segment group
      if good_mask[j] and segment_mask[j] == 0 and angleClose(angles[i], angles[j], angle_threshold):
          segment_mask[j] = segment_idx
    # Iterate to next group
    segment_idx += 1
  
  return segment_mask # segments

def chooseBestSegments(segments, line_mags):
  num_segments = segments.max() # 1-indexed, 0 is a masked/bad segment
  segment_mags = np.zeros(num_segments+1)
  for i in range(1, num_segments+1):
    num_in_segment = np.sum(segments == i)
    if num_in_segment < 4:
      # Need at least 4 lines in a segment
      segment_mags[i] = 0
    else:
      # Get average line gradient magnitude for that segment
      segment_mags[i] = np.sum(line_mags[segments == i])/num_in_segment
          
  # print("num:",num_segments)
  # print("mags:",segment_mags)
  order = np.argsort(segment_mags)[::-1]
  return order[:2] # Top two segments only