from PIL import Image
import csv
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

feature_counts = []

"""
def parseCSV(csvFile):
  with open(csvFile, 'r') as f:
    for row in csv.DictReader(f):
      readImage(row['image'] + '.jpeg')
"""

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

if __name__ == '__main__':
  sources = ['/media/aweeeezy/aweeeezy_hd/home/aweeeezy/bin/python/Image-Classifcation/files/images/train', '/media/aweeeezy/aweeeezy_hd/home/aweeeezy/bin/python/Image-Classifcation/files/images/test']
  readImages(sources)
  plotStatistics()
