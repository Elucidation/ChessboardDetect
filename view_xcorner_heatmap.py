import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays

# filepath = 'input/img_01.jpg'
filepath = 'testB.jpg'
img = PIL.Image.open(filepath).convert('L') # Grayscale full image with chessboard
img_width, img_height = img.size
aspect_ratio = min(200.0/img_width, 200.0/img_height)
new_width, new_height = ((np.array(img.size) * aspect_ratio)).astype(int)
img = img.resize((new_width,new_height), resample=PIL.Image.BILINEAR)
img = np.array(img)
heatmap[i] = prediction['probabilities'][1] # Probability is saddle
winsize=5

print(img.shape)


print(img.shape[0]-winsize-1 - (winsize+1))
print(img.shape[1]-winsize-1 - (winsize+1))

new_size = np.array([img.shape[0] - 2*winsize - 1, img.shape[1] - 2*winsize - 1])

print(new_size)

heatmap = np.load('heatmap.npy')
heatmap = np.reshape(heatmap, new_size)
print(heatmap)
# heatmap[heatmap<0.5] = 0

plt.imshow(heatmap)
plt.show()