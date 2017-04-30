import time
import sys
import itertools
import numpy as np
import pickle
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize
from scipy.misc import imread
from PIL import Image
from os import listdir
from multiprocessing import Pool

def resizeImage(args):
  image = Image.open(args[0]+ "/" + args[1])
  dim = args[2].split('x')
  image = image.resize((int(dim[0]), int(dim[1])), Image.ANTIALIAS)
  image.save('./data/train_resized/' + args[2] + '/' + args[1])

def performPCA(source, num_components, chunk_size):
  image_paths = listdir(source)
  size, images = 0, []
  n_chunks = len(image_paths)//chunk_size
  pca = IncrementalPCA(n_components=num_components, batch_size=chunk_size)
  for i in range(n_chunks):
    print('Chunk:', i, '\tIndex:', i * chunk_size + size)
    while size < chunk_size:
      images.append(imread(source+image_paths[i * chunk_size + size]).flatten())
      size += 1

    pca.partial_fit(np.asarray(images))
    size, images = 0, []

    if i == n_chunks - 1:
      i += 1
      while i * chunk_size + size < len(image_paths):
        images.append(imread(source+image_paths[i * chunk_size + size]).flatten())
        size += 1
      pca.partial_fit(np.asarray(images))


  print("\nExplained variance ratios: {0}".format(pca.explained_variance_ratio_))
  print("Sum of variance captured by components: {0}\n".format(sum(pca.explained_variance_ratio_)))

  xTransformed = None
  for i in range(n_chunks):
    while size < chunk_size:
      images.append(imread(source+image_paths[i * chunk_size + size]).flatten())
      size += 1
    print('chunk:', i, 'index:', i * chunk_size + size)
    transformed = pca.transform(np.asarray(images))
    if xTransformed is None:
      xTransformed = transformed
    else:
      xTransformed = np.vstack((xTransformed, transformed))
    size, images = 0, []

    if i == n_chunks - 1:
      i += 1
      while i * chunk_size + size < len(image_paths):
        images.append(imread(source+image_paths[i * chunk_size + size]).flatten())
        size += 1
      transformed = pca.transform(np.asarray(images))
      xTransformed = np.vstack((xTransformed, transformed))

  print("\nTransformed matrix shape:", xTransformed.shape)
  return xTransformed


if __name__ == "__main__":
  # Resizes all images in `source` path to `new_size`
  """
  source = './data/train_orig/'
  new_size = '32x32'
  pool = Pool()
  start = time.time()
  pool.map(resizeImage, zip(itertools.repeat(source), listdir(source), itertools.repeat(new_size)))
  print("Resized images in {0} seconds".format(time.time() - start))
  """

  # Performs incremental PCA on all images in `source` path...
  # Dumps binary file with a numpy matrix of transformed training data
  """
  source = './data/train_resized/32x32/'
  num_components = 8
  chunk_size = 1000
  start = time.time()
  transformedMatrix = performPCA(source, num_components, chunk_size)
  print("Performed PCA in {0} seconds".format(time.time() - start))
  normed_matrix = normalize(transformedMatrix)
  pickle.dump(normed_matrix, open('data/reduced_data.bin', 'wb'))
  """
