import matplotlib.pyplot as plt
import json
import numpy as np

data = {'full': {'svm': {}, 'knn': {}, 'xgb': {}},
        'limited': {'svm': {}, 'knn': {}, 'xgb': {}},
        'binary': {'svm': {}, 'knn': {}, 'xgb': {}}}

with open('../data/results.dat', 'r') as f:
  lines = [l.strip('\n') for l in f.readlines()]

for i, l in enumerate(lines):
  if i % 2 == 0:
    header = l.split(',')
    header[1] = ','.join(header[1:])
    next_line = [float(l) for l in lines[i+1].split(',')[:-1]]
    d = json.loads(header[1].replace("'", "\""))
    if 'kernel' in d:
      data[header[0]]['svm'][str(d)] = next_line
    elif 'n_neighbors' in d:
      data[header[0]]['knn'][str(d)] = next_line
    else:
      data[header[0]]['xgb'][str(d)] = next_line

def keyWithMaxValue(d, i):
  k = list(d.keys())
  v = list(d.values())
  return k[v.index(max(v, key=lambda x: x[i]))]

def iterateVariations():
  print("Printing model performance by metric for each model for each data subset...\nFormat: metric name \t parameters \t metric value\n")
  for s in ['full', 'limited', 'binary']:
    print("'{0}'".format(s))
    for m in ['KNN', 'SVM', 'XGB']:
      print("\t\t{0}:".format(m))
      for metric, index in [('Accuracy', 2), ('Precision', 3), ('Recall', 4),
          ('F-score', 5)]:
        key = keyWithMaxValue(data[s][m.lower()], index)
        value = data[s][m.lower()][str(key)][index]
        print("\t\t\t{0}\t{1}\t{2}".format(metric, key, value))

iterateVariations()

# Plotting training / prediction time for all models
"""
bin_knn_time_x = [i[0] for i in data['binary']['knn'].values()]
bin_knn_time_y = [i[1] for i in data['binary']['knn'].values()]
limited_knn_time_x = [i[0] for i in data['limited']['knn'].values()]
limited_knn_time_y = [i[1] for i in data['limited']['knn'].values()]
full_knn_time_x = [i[0] for i in data['full']['knn'].values()]
full_knn_time_y = [i[1] for i in data['full']['knn'].values()]

bin_svm_time_x = [i[0] for i in data['binary']['svm'].values()]
bin_svm_time_y = [i[1] for i in data['binary']['svm'].values()]
limited_svm_time_x = [i[0] for i in data['limited']['svm'].values()]
limited_svm_time_y = [i[1] for i in data['limited']['svm'].values()]
full_svm_time_x = [i[0] for i in data['full']['svm'].values()]
full_svm_time_y = [i[1] for i in data['full']['svm'].values()]

bin_xgb_time_x = [i[0] for i in data['binary']['xgb'].values()]
bin_xgb_time_y = [i[1] for i in data['binary']['xgb'].values()]
limited_xgb_time_x = [i[0] for i in data['limited']['xgb'].values()]
limited_xgb_time_y = [i[1] for i in data['limited']['xgb'].values()]
full_xgb_time_x = [i[0] for i in data['full']['xgb'].values()]
full_xgb_time_y = [i[1] for i in data['full']['xgb'].values()]

fig, ax = plt.subplots()

ax.scatter(bin_knn_time_x, bin_knn_time_y, color='b')
ax.scatter(limited_knn_time_x, limited_knn_time_y, color='g')
ax.scatter(full_knn_time_x, full_knn_time_y, color='r')

ax.scatter(bin_svm_time_x, bin_svm_time_y, color='b', marker='s')
ax.scatter(limited_svm_time_x, limited_svm_time_y, color='g', marker='s')
ax.scatter(full_svm_time_x, full_svm_time_y, color='r', marker='s')

ax.scatter(bin_xgb_time_x, bin_xgb_time_y, color='b', marker='^')
ax.scatter(limited_xgb_time_x, limited_xgb_time_y, color='g', marker='^')
ax.scatter(full_xgb_time_x, full_xgb_time_y, color='r', marker='^')

import matplotlib.patches as patches
from matplotlib.lines import Line2D

red = patches.Patch(color='red', label='Full Dataset')
green = patches.Patch(color='green', label='Limited Dataset')
blue = patches.Patch(color='blue', label='Binary Dataset')

knn_dot = Line2D([], [], markersize='15', color='white', label='KNN', marker='o')
svm_square = Line2D([], [], markersize='15', color='white', label='SVM', marker='s')
xgb_carrot = Line2D([], [], markersize='15', color='white', label='XGBoost', marker='^')

plt.legend(handles=[red, green, blue, knn_dot, svm_square, xgb_carrot])

plt.xlabel("Train Time")
plt.ylabel("Predict Time")
plt.title("Training/Predicting Time by Data Subset and Model Type")
plt.show()
"""


"""

### KNN results sorted by n_neighbors -- BINARY
# BINARY
s = sorted(data['binary']['knn'], key=lambda k: list(json.loads(k.replace("'", "\"")).values())[0])
print("\nKNN results sorted by n_neighbors -- BINARY")
for k in s:
  print(data['binary']['knn'][k])
bin_knn_time_x = [i[0] for i in s]
bin_knn_time_y = [i[1] for i in s]

# LIMITED
s = sorted(data['limited']['knn'], key=lambda k: list(json.loads(k.replace("'", "\"")).values())[0])
print("\nKNN results sorted by n_neighbors -- LIMITED")
for k in s:
  print(data['limited']['knn'][k])
limited_knn_time_x = [i[0] for i in s]
limited_knn_time_y = [i[1] for i in s]

# FULL
s = sorted(data['full']['knn'], key=lambda k: list(json.loads(k.replace("'", "\"")).values())[0])
print("\nKNN results sorted by n_neighbors -- FULL")
for k in s:
  print(data['full']['knn'][k])
full_knn_time_x = [i[0] for i in s]
full_knn_time_y = [i[1] for i in s]
"""
