from os import listdir
import time
from multiprocessing import Pool, Manager, Lock
from PIL import Image
import itertools
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import pickle

"""
  Reads images sequentially to determine the distribution of pixel counts.
"""
def readImages(source):
  feature_counts = []
  for image in listdir(source):
    pixels = np.array(Image.open(source + image))
    features = pixels.shape[0] * pixels.shape[1]
    feature_counts.append(features)
  return feature_counts

"""
  Reads an image to determine the distribution of pixel counts -- meant to be
  used with a multiprocessing pool for parallel execution.
"""
def readImagesParallel(args):
  pixels = np.array(Image.open(args[0] + args[1]))
  features = pixels.shape[0] * pixels.shape[1]
  lock.acquire()
  args[2].append(features)
  lock.release()

"""
  Plots two histograms -- one with a number of bins equal to the square root of
  the number of images being processed, the other with 10 bins -- and a
  horizontal bar chart showing the pixel count over the frequency count. Note:
  the horizontal bar chart is kinda fucked up...I can't remember how the xticks
  were formatted in order to produce the png images in the repository so just
  play with it for a while to see if you can get it to look nice.
"""
def plotStatistics():
  plt.hist(feature_counts, round(sqrt(len(feature_counts))), facecolor='green', alpha=0.75)
  plt.xlabel('# Pixels')
  plt.ylabel('Frequency')
  plt.grid(True)
  plt.show()

  plt.hist(feature_counts, 10, facecolor='green', alpha=0.75)
  plt.xlabel('# Pixels')
  plt.ylabel('Frequency')
  plt.grid(True)
  plt.show()

  fig, ax = plt.subplots()
  sorted_data = sorted(feature_counts)
  plt.barh(np.arange(len(sorted_data)), sorted_data, height=1, alpha=0.5)
  plt.xticks(np.arange(max([sorted_data.count(s) for s in set(sorted_data)])),
      [sorted_data.count(s) for s in set(sorted_data)])
  plt.yticks(np.arange(len(sorted_data))+.5, set(sorted_data))
  plt.ylim(0,len(set(sorted_data)))
  #for i,v in enumerate(sorted_data):
  #  ax.text(v+(500*(len(str(v)))), i+.4, str(sorted_data.count(v)))
  plt.tight_layout()
  plt.xlabel('Frequency')
  plt.ylabel('# Pixels')
  plt.show()


if __name__ == '__main__':
  lock = Lock()
  source = '../data/train_originals/'
  p = Pool()
  m = Manager()
  l = m.list()
  start = time.time()
  p.map(readImagesParallel, zip(itertools.repeat(source), listdir(source), itertools.repeat(l)))
  print("Image analysis took {0} seconds".format(time.time()- start))
  pickle.dump(list(l), open('../data/statistics/feature_counts.bin', 'wb'))
  #feature_counts = loadData()
  plotStatistics()
