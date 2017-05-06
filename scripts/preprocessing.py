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

"""
  See main function at bottom for usage. Meant to be used with 'multiprocessing'.
"""
def resizeImage(args):
  image = Image.open(args[0]+ "/" + args[2])
  dim = args[3].split('x')
  image = image.resize((int(dim[0]), int(dim[1])), Image.ANTIALIAS)
  image.save(args[1] + args[2])

"""
  See main function at bottom for usage. Reads in 'chunk_size' segments of the
  training images directory and performs incremental PCA on them in order to
  reduce the dimensionality of the images to 'num_components'. Returns a numpy
  matrix with dimensions (sample_size X 'num_components').
"""
def performPCA(source, num_components, chunk_size):
  image_paths = sorted(listdir(source), key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
  size, images = 0, []
  n_chunks = len(image_paths)//chunk_size
  pca = IncrementalPCA(n_components=num_components, batch_size=chunk_size)

  # Read in all images and do a partial fit on the PCA model.
  for i in range(n_chunks):
    print 'Chunk:', i, 'Index:', i * chunk_size + size
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

  # Only works with Python 3
  #print("\nExplained variance ratios: {0}".format(pca.explained_variance_ratio_))
  #print("Sum of variance captured by components: {0}\n".format(sum(pca.explained_variance_ratio_)))

  xTransformed = None

  # Read in all images again and transform them using the PCA model.
  for i in range(n_chunks):
    while size < chunk_size:
      images.append(imread(source+image_paths[i * chunk_size + size]).flatten())
      size += 1
    print 'Chunk:', i, 'index:', i * chunk_size + size
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

  print "\nTransformed matrix shape:", xTransformed.shape
  return xTransformed


if __name__ == "__main__":
  """
    !!! Make sure 'source' and 'target are directories that aleady exist. When
    running performePCA(), make sure 'chunk_size' is less than the number of
    images you're processing. If you're running into memory issues, try
    adjusting 'chunk_size' to a smaller value.
  """

  # Resizes images in `source` to `new_size` and save them to 'target'.
  # Uncomment this block to automatically resize images when executing this script.
  """
  source = '../data/train_originals/'
  target = '../data/train_resized/32x32/'
  new_size = '32x32'
  pool = Pool()
  start = time.time()
  pool.map(resizeImage, zip(itertools.repeat(source), itertools.repeat(target), listdir(source), itertools.repeat(new_size)))
  print "Resized images in {0} seconds".format(time.time() - start)
  """

  # Performs incremental PCA on all images in `source` path. Dumps binary file
  # with a numpy matrix of transformed training data. Uncomment this block to
  # automatically reduce dimensionality when executing this script.
  source = '../data/train_resized/32x32/'
  num_components = 8
  chunk_size = 1000
  start = time.time()
  transformedMatrix = performPCA(source, num_components, chunk_size)
  print "Performed PCA in {0} seconds".format(time.time() - start)
  normed_matrix = normalize(transformedMatrix)
  pickle.dump(normed_matrix, open('../data/reduced_data.bin', 'wb'))
