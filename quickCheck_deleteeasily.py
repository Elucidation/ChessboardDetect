import numpy as np

np_dataset = np.load('dataset_5.npz')
features = np_dataset['features']
labels = np_dataset['labels']

print(len(labels))
print(sum(labels))