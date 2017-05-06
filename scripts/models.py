import time
import pandas
import pickle
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

"""
  See main function at bottom for usage. Returns a data dictionary where the
  keys are 'full', 'limited', and 'binary'. The value for each key is another
  dictionary where the keys of that dictionary are 'X_train', 'X_test',
  'y_train', and 'y_test'. The values for these nested dictionaries are numpy
  matrices/vectors for the properly partitioned training/testing data/labels.
"""
def prepareData():
  # Read in reduced data and make copies that will be mutated.
  full_PCA_data = pickle.load(open('../data/reduced_data.bin', 'rb'))
  limited_PCA_data = full_PCA_data
  binary_PCA_data = full_PCA_data

  # Read in labels from CSV for training.
  df = pandas.read_csv('trainLabels.csv')
  temp_labels = df.as_matrix()

  full_labels = df.as_matrix()[:,1]
  limited_labels = full_labels
  binary_labels = full_labels

  # Creates 'limited' and 'binary' data which have fewer '0' class objects so
  # that the number of examples for each class is more balanced...'limited' has
  # a number of '0' class objects equal to the number of class '2' objects
  # while 'binary' has the same number of '0' class objects and additionally
  # removes all '1', '3', and '4' class objects.

  # Builds sets of row indices to be included in the final data subsets for
  # 'limited' and 'binary'.
  i, counter, limited, binary = 0, 0, set(), set()
  for i, l in enumerate(temp_labels):
    if counter < 5292 and l[1] == 0:
      limited.add(i)
      binary.add(i)
      counter += 1
    elif l[1] == 2:
      limited.add(i)
      binary.add(i)
    elif l[1] != 0:
      limited.add(i)

  # Removes row indices that do not belong in the 'limited' data/label subsets.
  i, j = 0, 0
  while len(limited_PCA_data) > len(limited):
    if j not in limited:
      limited_PCA_data = np.delete(limited_PCA_data, i, 0)
      limited_labels = np.delete(limited_labels, i, 0)
      i -= 1
    i += 1
    j += 1

  # Removes row indices that do not belong in the 'binary' data/label subsets.
  i, j = 0, 0
  while len(binary_PCA_data) > len(binary):
    if j not in binary:
      binary_PCA_data = np.delete(binary_PCA_data, i, 0)
      binary_labels = np.delete(binary_labels, i, 0)
      i -= 1
    i += 1
    j += 1

  print "Shapes:\n\tfull_PCA_data: {0}\n\tfull_labels: {1}\n\tlimited_PCA_data: {2}\n\tlimited_labels {3}\n\tbinary_PCA_data: {4}\n\tbinary_labels: {5}".format(*[e.shape for e in [full_PCA_data, full_labels, limited_PCA_data, limited_labels, binary_PCA_data, binary_labels]])

  # 'test_size' and 'seed' for splitting so that splits are the same for each run.
  seed = 7
  size = 0.3

  # Data and their appropriate labels for each training configuration.
  configurations = [
    ('full', full_PCA_data, full_labels),
    ('limited', limited_PCA_data, limited_labels),
    ('binary', binary_PCA_data, binary_labels)
  ]

  # Initialized data dictionary that is returned by this funciton.
  data = {
    'full': {}, 'limited': {}, 'binary': {}
  }

  # Populate data dictionary with appropriate data/label subsets.
  for c in configurations:
    X_train, X_test, y_train, y_test = train_test_split(*c[1:], test_size=size, random_state=seed)
    data[c[0]]['X_train'] = X_train
    data[c[0]]['X_test'] = X_test
    data[c[0]]['y_train'] = y_train
    data[c[0]]['y_test'] = y_test

  # Convert labels from strings to ints.
  for _, v in data.items():
    v['y_train'], v['y_test'] = v['y_train'].astype(int), v['y_test'].astype(int)

  return data

"""
  See main function at bottom for usage. Takes in a key for the data/label
  subsets and a dictionary of model parameters, then trains and predicts with
  each type of model.
"""
def trainModels(data, params):
  f = open(out_file, 'w')
  def trainKNN(data_subset):
    f.write('\nTraining KNN:'+'\n')

    X_train = data[data_subset]['X_train']
    X_test = data[data_subset]['X_test']
    y_train = data[data_subset]['y_train']
    y_test = data[data_subset]['y_test']

    for p in params['knn']:
      header = "@ subset: {0}, params: {1}".format(data_subset, p)
      f.write('\n'+header+'\n')

      n_neighbors = p['n_neighbors']

      model = KNeighborsClassifier(n_neighbors=n_neighbors)

      start = time.time()
      model.fit(X_train, y_train)
      elapsed_train = time.time() - start

      y_pred = model.predict(X_test).astype(int)
      elapsed_predict = time.time() - start

      accuracy = accuracy_score(y_test, y_pred)
      precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, pos_label=2, average='weighted')

      print("\n{5}\nKNN with {0} neighbors on data subset {1} trained in {2} seconds and predicted in {3} seconds with an accuracy of {4}\n".format(n_neighbors, data_subset, elapsed_train, elapsed_predict, accuracy, header))

      f.write(str(elapsed_train) + ', ' + str(elapsed_predict) + str(accuracy)+ ', ' + str(precision)+ ', ' + str(recall )+ ', ' + str(fscore )+ ', ' + str(support))

  def trainSVM(data_subset):
    f.write('\nTraining SVM:'+'\n')

    X_train = data[data_subset]['X_train']
    X_test = data[data_subset]['X_test']
    y_train = data[data_subset]['y_train']
    y_test = data[data_subset]['y_test']

    for p in params['svm']:
      header = "@ subset: {0}, params: {1}".format(data_subset, p)
      f.write('\n'+header+'\n')

      kernel = p['kernel']
      gamma = p['gamma']

      model = svm.SVC(kernel=kernel, gamma=gamma)

      start = time.time()
      model.fit(X_train, y_train)
      elapsed_train = time.time() - start

      y_pred = model.predict(X_test).astype(int)
      elapsed_predict = time.time() - start

      accuracy = accuracy_score(y_test, y_pred)
      precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, pos_label=2, average='weighted')

      print("\n{5}\nSVM with {0} kernel and {6} gamma on data subset {1} trained in {2} seconds and predicted in {3} seconds with an accuracy of {4}\n".format(kernel, data_subset, elapsed_train, elapsed_predict, accuracy, header, gamma))

      f.write(str(elapsed_train) + ', ' + str(elapsed_predict) + str(accuracy)+ ', ' + str(precision)+ ', ' + str(recall )+ ', ' + str(fscore )+ ', ' + str(support))

  def trainXGB(data_subset):
    f.write('\nTraining XGB:'+'\n')

    X_train = data[data_subset]['X_train']
    X_test = data[data_subset]['X_test']
    y_train = data[data_subset]['y_train']
    y_test = data[data_subset]['y_test']

    for p in params['xgboost']:
      if data_subset != 'binary' and p['objective'] == 'binary:logistic':
        print("Skip using non-binary data with XGB binary:logistic objective")
        continue
      if data_subset == 'binary' and p['objective'] != 'binary:logistic':
        print("Skip using binary data with XGB multi:* objective")
        continue

      header = "@ subset: {0}, params: {1}".format(data_subset, p)
      f.write('\n'+header+'\n')

      objective = p['objective']
      max_depth = p['max_depth']
      try:
        n_estimators= p['n_estimators']
      except KeyError as e:
        n_estimators= 100

      model = XGBClassifier(objective=objective, max_depth=max_depth,
        n_estimators=n_estimators)

      start = time.time()
      model.fit(X_train, y_train)
      elapsed_train = time.time() - start

      y_pred = model.predict(X_test).astype(int)
      elapsed_predict = time.time() - start

      accuracy = accuracy_score(y_test, y_pred)
      precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, pos_label=2, average='weighted')

      print("\n{5}\nXGB with {0} objective, {6} max_depth, {7} n_estimators on data subset {1} trained in {2} seconds and predicted in {3} seconds with an accuracy of {4}\n".format(objective, data_subset, elapsed_train, elapsed_predict, accuracy, header, max_depth, n_estimators))

      f.write(str(elapsed_train) + ', ' + str(elapsed_predict) + str(accuracy)+ ', ' + str(precision)+ ', ' + str(recall )+ ', ' + str(fscore )+ ', ' + str(support))

  # Iterate over all the data/label subsets and then train and predict with
  # each type of model.
  for k, v in data.items():
    print("\nTraining {0} subset on KNN classifier\n".format(k))
    trainKNN(k)
    print("\nTraining {0} subset on XGB classifier\n".format(k))
    trainXGB(k)
    print("\nTraining {0} subset on SVM classifier\n".format(k))
    trainSVM(k)

  f.close()

if __name__ == '__main__':
  # Dictionary of parameters keyed by the type of model. Values are lists of
  # parameter sets.
  params = {
      'knn': [{'n_neighbors': 3}, {'n_neighbors': 5},
        {'n_neighbors': 10}, {'n_neighbors': 25}, {'n_neighbors': 50}],
    'xgboost': [
      {'objective': 'multi:softmax', 'max_depth': 3, 'n_estimators': 100},
      {'objective': 'multi:softmax', 'max_depth': 6, 'n_estimators': 100},
      {'objective': 'multi:softmax', 'max_depth': 10, 'n_estimators': 100},
      {'objective': 'multi:softmax', 'max_depth': 3, 'n_estimators': 1000},
      {'objective': 'multi:softmax', 'max_depth': 6, 'n_estimators': 1000},
      {'objective': 'multi:softmax', 'max_depth': 10, 'n_estimators': 1000},
      {'objective': 'multi:prob', 'max_depth': 3, 'n_estimators': 100},
      {'objective': 'multi:prob', 'max_depth': 6, 'n_estimators': 100},
      {'objective': 'multi:prob', 'max_depth': 10, 'n_estimators': 100},
      {'objective': 'multi:prob', 'max_depth': 3, 'n_estimators': 1000},
      {'objective': 'multi:prob', 'max_depth': 6, 'n_estimators': 1000},
      {'objective': 'multi:prob', 'max_depth': 10, 'n_estimators': 1000},
      {'objective': 'binary:logistic', 'max_depth': 3, 'n_estimators': 100},
      {'objective': 'binary:logistic', 'max_depth': 6, 'n_estimators': 100},
      {'objective': 'binary:logistic', 'max_depth': 10, 'n_estimators': 100},
      {'objective': 'binary:logistic', 'max_depth': 3, 'n_estimators': 1000},
      {'objective': 'binary:logistic', 'max_depth': 6, 'n_estimators': 1000},
      {'objective': 'binary:logistic', 'max_depth': 10, 'n_estimators': 1000},
    ],
    'svm': [
      {'kernel': 'rbf', 'gamma': 0.1},
      {'kernel': 'rbf', 'gamma': 0.5},
      {'kernel': 'rbf', 'gamma': 0.9},
      {'kernel': 'poly', 'gamma': 0.1, 'degree': 3},
      {'kernel': 'poly', 'gamma': 0.5, 'degree': 3},
      {'kernel': 'poly', 'gamma': 0.9, 'degree': 3},
      {'kernel': 'poly', 'gamma': 0.1, 'degree': 10},
      {'kernel': 'poly', 'gamma': 0.5, 'degree': 10},
      {'kernel': 'poly', 'gamma': 0.9, 'degree': 10}
    ]
  }

  out_file = '../data/results.dat'

  start = time.time()
  data = prepareData()
  trainModels(data, params)
  print("\nEntire training took {0} seconds".format(time.time() - start))
