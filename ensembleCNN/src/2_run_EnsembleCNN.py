#!/usr/bin/env python
# coding: utf-8

import json
import sys
import time
import pprint
import datetime
import fcsparser
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import LogNorm
from pathlib import Path
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count
from sklearn import preprocessing
import sklearn.model_selection
import sklearn

import statistics
import pickle
import os
import logging

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        logfilename = "./log/2_bin" + sys.argv[1] + ".logfile.log"
        if os.path.exists(logfilename):
            os.remove(logfilename)
        self.log = open(logfilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()

t1 = datetime.datetime.now()
t1p = time.process_time()
print("TSTAMP A : " + str(t1))

## vars
data_directory = '/prj0133/ResearchData/2020FlowcatData/flowCatData/decCLL-9F/'
if not os.path.exists(data_directory):
    data_directory = '/athena/marchionnilab/scratch/lab_data/wikum/2020FlowcatData/flowCatData/decCLL-9F/'
save_dir = '../out/'

with open('../obj/good_data.json') as f:
    good_data = json.load(f)

print("good data length: " + str(len(good_data)))

with open('../obj/good_data_vars.obj', 'rb') as f:
    num_bins, num_columns, total_histograms, marker_panels, the_marker_panels, curated_marker_panels, data_cohorts, data_diagnoses = pickle.load(f)

ids = [x['id'] for x in good_data]
groups = [x['cohort'] for x in good_data]

y = np.asarray(groups)
ids = np.asarray(ids)

if len(sys.argv) <= 1:
    num_bins = 50
else:
    num_bins = int(sys.argv[1])

print("num_bins = " + str(num_bins))

## load X

X = np.load(save_dir + "X_" + str(num_bins) + ".npy")
print(X.shape)
num_rows = len(good_data)
num_columns = int(X.shape[0]/len(good_data))
if(X.shape[0] == num_rows):
    print("No need to reshape X")
else:
    X = X.reshape(num_rows, num_columns)
    print("reshaping...")
    print(X.shape)

## select training data

with open('../obj/train_test_ids.obj', 'rb') as f:
    train_ids, test_ids = pickle.load(f)

y_train = y[train_ids, ]
y_test = y[test_ids, ]

## run with train ( CNN ) tune (RF) split used for the results
with open('../../../ensembleCNN/run1/obj/4_cnn_rf_splits.obj', 'rb') as f:
    train_ids1, tune_ids1, train_ids2, tune_ids2, train_ids3, tune_ids3 = pickle.load(f)

def get_matching_indices(x, y): ## x is reference list, y is query
    arr1 = np.array(x)
    arr2 = np.array(y)
    res = np.where(np.isin(arr1, arr2))[0]
    return res

train_ids1 = get_matching_indices(ids, train_ids2)
train_ids2 = get_matching_indices(ids, tune_ids2)

y_train1 = y[train_ids1, ]
y_train2 = y[train_ids2, ]

# train_ids1, train_ids2, y_train1, y_train2 = sklearn.model_selection.train_test_split(train_ids, y_train, train_size=0.5)

print("CNNs: ")
print(pd.value_counts(y[train_ids1, ]))
print("RF: ")
print(pd.value_counts(y[train_ids2, ]))

t2 = datetime.datetime.now()
t2p = time.process_time()
print("TSTAMP B : " + str(t2))

print("time elapsed: " + str(t2-t1))
print("Process time: B-A = ", t2p-t1p)


## fit CNN
import EnsembleCNN

classifier = EnsembleCNN.EnsembleCNNClassifier()
classifier.num_bins = num_bins
classifier.set_model_folder("../output/b" + str(num_bins) + "_models")

classifier.fit_CNNs(X[train_ids1, ], y_train1, batch_size=256, epochs=20, validation_split=0.33)

if(len(classifier.CNN_models) < classifier.num_features):
    classifier.reload_CNNs()

Z_train1 = classifier.get_CNN_outputs(X[train_ids1, ])
Z_train2 = classifier.get_CNN_outputs(X[train_ids2, ])
Z_test = classifier.get_CNN_outputs(X[test_ids, ])

t3 = datetime.datetime.now()
t3p = time.process_time()
print("TSTAMP C : " + str(t3))

print("time elapsed: " + str(t3-t2))

print("Process time: C-B = ", t3p-t2p)


print("Fitting RF....")

classifier.fit_RF(Z_train2, y[train_ids2, ], n_estimators=1000, max_leaf_nodes=50, rand_seed=1000)

print("Training results : ")

print(sklearn.metrics.classification_report(y[train_ids2, ], classifier.get_RF_outputs(Z_train2)))

y_pred = classifier.get_RF_outputs(Z_test)

with open('../output/y_preds.obj', 'wb') as f:
    pickle.dump([y_test, y_pred], f)

print("Testing results : ")

print(sklearn.metrics.classification_report(y_test, y_pred))

t4 = datetime.datetime.now()
t4p = time.process_time()
print("TSTAMP D : " + str(t4))

print("time elapsed: " + str(t4-t3))
print("Process time: D-C = ", t4p-t3p)

print("---------")

print("total time elapsed: " + str(t4-t1))
print("total process time: D-A = ", t4p-t1p)

print("done.")
