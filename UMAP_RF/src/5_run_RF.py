import json
import sys
import time
import pprint
import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import LogNorm
from pathlib import Path
import numpy.random

#Machine learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
import sklearn

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
import tensorflow

import statistics
import pickle
from timeit import default_timer as timer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import os
import fastcluster
import seaborn as sns
import joblib
import random
from sklearn import preprocessing
random.seed(1)


import itertools
import shap
import umap
import matplotlib

# user options
data_directory = '/Volumes/USB30FD/ResearchData/2020FlowcatData/flowCatData/decCLL-9F/'
obj_directory = '/Users/wikumwcm/WORK/LAB/prj0133/wikum/ensembleCNN/run1/obj/'
model_dir = '/Volumes/easystore/LAB/prj0133/wikum/ensembleCNN/run3/output/'
X_directory = '/Volumes/easystore/LAB/jupyter/testpy2/obj'
output_dir = './output/'

with open(obj_directory + '/good_data.json') as f:
    good_data = json.load(f)

print("good data length: " + str(len(good_data)))

with open(obj_directory + '/1_vars.obj', 'rb') as f:
    num_bins, num_columns, total_histograms, marker_panels, the_marker_panels, curated_marker_panels, data_cohorts, data_diagnoses = pickle.load(f)

print("-")

print(num_bins)
print(num_columns)
print(total_histograms)
print(curated_marker_panels)
print(pd.value_counts(data_cohorts))
print(pd.value_counts(data_diagnoses))

with open(obj_directory + '1_vars.obj', 'rb') as f:
    num_bins, num_columns, total_histograms, marker_panels, the_marker_panels, curated_marker_panels, data_cohorts, data_diagnoses = pickle.load(f)

with open(obj_directory + 'train_tune_test_dict.obj', 'rb') as f:
    sel_train, sel_tune, sel_test = pickle.load(f)

with open(obj_directory + '4_cnn_rf_splits.obj', 'rb') as f:
    train_ids1, tune_ids1, train_ids2, tune_ids2, train_ids3, tune_ids3 = pickle.load(f)

print(num_bins)
print(num_columns)
print(total_histograms)
print(curated_marker_panels)
print(pd.value_counts(data_cohorts))
print(pd.value_counts(data_diagnoses))


y = []
ids = []
for i, data_x in enumerate(good_data):
    y.append(data_x['cohort'])
    ids.append(data_x['id'])

y0 = y
y = np.asarray(y)
ids = np.asarray(ids)

def dicttolist(x):
    y = []
    for cx in x.keys():
        y = y + x[cx]
    y = (list(y))
    return y

## test_ids = dicttolist(sel_test)

def get_matching_indices(x, y): ## x is reference list, y is query
    arr1 = np.array(x)
    arr2 = np.array(y)
    res = np.where(np.isin(arr1, arr2))[0]
    return res

## train_ids = get_matching_indices(ids, train_ids2)
## tune_ids = get_matching_indices(ids, tune_ids2)


train_ids = train_ids2
tune_ids = tune_ids2
test_ids = [ids[xi] for xi in dicttolist(sel_test)]

print(train_ids[0:5])
print(tune_ids[0:5])
print(test_ids[0:5])

if False:
    ids = [x['id'] for x in good_data]
    groups = [x['cohort'] for x in good_data]

    y_dict = dict(zip(ids, groups))

    cohorts = []
    for yi in groups:
        if(yi not in cohorts):
            cohorts.append(yi)

    cohort_dict = {}
    ctr = 0
    for ci in cohorts:
        di = []
        for i in range(0, len(groups)):
            if groups[i] == ci:
                di.append(i)

        cohort_dict[ci] = di
        ctr = ctr + len(di)

    with open("./output/train_test_ids.obj", "rb") as f:
        train_ids_all, test_ids = pickle.load(f)


    sample_dict = []
    for cl in ['LPL', 'MCL', 'PL', 'normal', 'MZL', 'CLL', 'MBL', 'HCL', 'FL']:
        random.seed(1)
        u = list(set(train_ids_all).intersection(cohort_dict[cl]))
        sample_dict = sample_dict + random.sample(u, 200)

    pd.value_counts([groups[i] for i in sample_dict])


with open('./output/hlist.obj', 'rb') as f:
    hlist = pickle.load(f)

hlist = [x.split(".")[0] for x in hlist]
## hlist = random.sample(hlist, len(hlist))

print(len(hlist))
print(hlist[0:10])

h_dict = dict(zip(hlist, range(len(hlist))))

hlist2 = [hlist[i] for i in range(len(hlist))]
hlist2.sort()
print(hlist2[0:10])
pd.value_counts([hlist[i] == hlist2[i] for i in range(len(hlist))])

print(len(train_ids))
train_h_ids = list(set(hlist).intersection(set(train_ids)))
print(len(train_h_ids))
print(train_h_ids[0:10])


print(len(tune_ids))
tune_h_ids = list(set(hlist).intersection(set(tune_ids)))
print(len(tune_h_ids))
print(tune_h_ids[0:10])


print(len(test_ids))
test_h_ids = list(set(hlist).intersection(set(test_ids)))
print(len(test_h_ids))
print(test_h_ids[0:10])

print(len(set(train_ids).intersection(set(tune_ids))))
print(len(set(train_ids).intersection(set(test_ids))))
print(len(set(tune_ids).intersection(set(test_ids))))


train_indices = [h_dict[x] for x in train_h_ids]
print(len(train_indices))
print(train_indices[0:5])
print("---")
tune_indices = [h_dict[x] for x in tune_h_ids]
print(len(tune_indices))
print(tune_indices[0:5])
print("---")
test_indices = [h_dict[x] for x in test_h_ids]
print(len(test_indices))
print(test_indices[0:5])


y_dict = dict(zip([x['id'] for x in good_data], [x['cohort'] for x in good_data]))
y_h_train = [y_dict[hlist[xi]] for xi in train_indices]
y_h_tune = [y_dict[hlist[xi]] for xi in tune_indices]
y_h_test = [y_dict[hlist[xi]] for xi in test_indices]


print(pd.value_counts(y_h_train))
print("---")
print(pd.value_counts(y_h_tune))
print("---")
print(pd.value_counts(y_h_test))


pid1 = "20230613231036" ## panel 0, 22 mil, balanced sampling
pid2 = "20230909230852" ## panel 1, 22 mil balanced sampling
pid3 = "20230615165901" ## panel 2, 22 mil (balanced sampling?)

X1 = np.transpose(np.load('./output/' + pid1 + '_H_sorted.npy'))
print(X1.shape)

X2 = np.transpose(np.load('./output/' + pid2 + '_H_sorted.npy'))
print(X2.shape)

X3 = np.transpose(np.load('./output/' + pid3 + '_H_sorted.npy'))
print(X3.shape)


## panel 1 + 2 + 3 (or 0 + 1 + 2)

X = np.concatenate((X1, X2, X3), axis=1)

print(X.shape)
print(X[train_indices, ].shape)
print(X[tune_indices, ].shape)
print(X[test_indices, ].shape)


R = RandomForestClassifier(n_estimators=5000, max_leaf_nodes=200, n_jobs=-1, verbose=1,
                           class_weight='balanced', random_state=1000)

y_true = y_h_train + y_h_tune
R.fit(X[train_indices + tune_indices, ], y_true)

y_true = y_h_test
y_pred = R.predict(X[test_indices, ])
ba = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
f = statistics.mean(sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)[2])

print(sklearn.metrics.classification_report(y_true, y_pred))
