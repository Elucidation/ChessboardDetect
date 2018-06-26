import tensorflow as tf 
import glob
import numpy as np
np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays


filenames = glob.glob('datasets/tfrecords/winsize_10_color/*.tfrecords')

c = 0
for filename in filenames:
  k = 0
  for record in tf.python_io.tf_record_iterator(filename):
     k += 1
  print('%s : %d entries' % (filename, k))
  c += k

print('Total : %d entries' % c)