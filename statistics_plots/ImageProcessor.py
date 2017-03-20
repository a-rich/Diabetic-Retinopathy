from os import listdir
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import pickle
from collections import OrderedDict

feature_counts = []

def readImages(sources):
  for path in sources:
    for image in listdir(path):
      pixels = np.array(Image.open(path + "/" + image))
      features = pixels.shape[0] * pixels.shape[1]
      feature_counts.append(features)

def plotStatistics():
  plt.hist(feature_counts, round(sqrt(len(feature_counts))), facecolor='green', alpha=0.75)
  plt.xlabel('# Pixels')
  plt.ylabel('Frequency')
  plt.grid(True)
  plt.show()

  """
  fig, ax = plt.subplots()
  ax.barh(np.array(range(len(sorted_stats))), values, height=1, alpha=0.5)
  plt.yticks(np.array(range(len(sorted_stats)))+0.5, [s[0] for s in sorted_stats])
  plt.ylim(0,31)
  for i,v in enumerate([s[1] for s in sorted_stats]):
    ax.text(v+(500*(len(str(v))*.70)), i+.2, str(v), horizontalalignment='right')
  plt.tight_layout()
  plt.xlabel('Frequency')
  plt.ylabel('# Pixels')
  plt.show()
  """

def loadData():
  feature_counts = pickle.load(open('./frequency_data', 'rb'))

def saveData():
  pickle.dump(open('./frequency_data', 'wb'))

if __name__ == '__main__':
  source_path = '/media/aweeeezy/aweeeezy_hd/home/aweeeezy/bin/python/Image-Classifcation/files/images/'
  sources = [source_path + 'train', source_path + 'test']
  readImages(sources)
  saveData()
  plotStatistics()
