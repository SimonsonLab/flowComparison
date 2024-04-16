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
from pathlib import Path
from timeit import default_timer as timer
import multiprocessing
from multiprocessing import Pool, cpu_count
import tensorflow as tf
from tensorflow.keras import layers
import umap
# from umap.parametric_umap import load_ParametricUMAP

import statistics
import pickle
import os
import logging
import random
import contextlib

RUN_ID = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

PANEL_NUM = int(sys.argv[1]) ## run panel 0, 1 or 2

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("./log/1_"+ RUN_ID + ".logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

## Code from Paul for reading FCS
def return_panel_path(panel, case):
    """
    This function finds the file path for FCS file for the panel for the case.
    Test: return_panel_path(the_marker_panels[0], good_data[0])
    """
    for file in case['filepaths']:
        if panel == file['fcs']['markers']:
            #print(file['fcs']['path'])
            return file['fcs']['path']

def return_FCS_data(panel, case, data_directory, curated=None):
    """
    Return the FCS file data corresponding the given antibody panel for the given case.
    Test: return_FCS_data(the_marker_panels[0], good_data[0])
    """
    relative_path = return_panel_path(panel, case)
    complete_path = data_directory + relative_path
    with open('./log/fcsread.txt', 'w') as f:
        with contextlib.redirect_stderr(f):
            meta, data = fcsparser.parse(complete_path, reformat_meta=True)

    channel_names = list(meta['_channel_names_'])
    data = pd.DataFrame(data)
    data.columns = channel_names
    if(curated == None):
        curated = panel
    data = data[curated]
    return data

## from David Ng
def convert_case_to_histo(df,N=32,display_range=np.array([[0,24],[0,28]])):
    # a = np.histogram2d(df['UMAP1'],df['UMAP2'],range=display_range,density=True,bins=(N,N))
    a = np.histogram2d(df[:, 0],df[:, 1],range=display_range,density=True,bins=(N,N))
    return a[0]/a[0].max()

sys.stdout = Logger()

print("============ RUN_ID:" + RUN_ID + "============")

with open('./RUNID.txt', 'a') as f:
    f.write(RUN_ID)

## vars
data_directory = '/prj0133/ResearchData/2020FlowcatData/flowCatData/decCLL-9F/'
if not os.path.exists(data_directory):
    data_directory = '/athena/marchionnilab/scratch/lab_data/wikum/2020FlowcatData/flowCatData/decCLL-9F/'
save_dir = '../out/'
## M = 22000000
M = 10000
sel_classes = ['LPL', 'MCL', 'PL', 'normal', 'MZL', 'CLL', 'MBL', 'HCL', 'FL'] ## no HCLv

with open('../obj/good_data.json') as f:
    good_data = json.load(f)

print("good data length: " + str(len(good_data)))

with open('../obj/good_data_vars.obj', 'rb') as f:
    num_bins, num_columns, total_histograms, marker_panels, the_marker_panels, curated_marker_panels, data_cohorts, data_diagnoses = pickle.load(f)


sel_panel = the_marker_panels[PANEL_NUM]
curated_panel = curated_marker_panels[PANEL_NUM]


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

with open("../obj/train_test_ids.obj", "rb") as f:
    train_ids_all, test_ids = pickle.load(f)

train_dict = {}
## no HCLv
for ci in sel_classes:
    print(ci)
    di = cohort_dict[ci]
    print(len(di))
    gi = list(set(di).intersection(set(train_ids_all)))
    print(len(gi))
    train_dict[ci] = gi

## set size of umap
xm = math.floor(M/len(train_dict)) ## events sampled by each subtype in total

xn_dict = {} ## events to be sampled from each case depending on the subtype
for cl in train_dict.keys():
    xn_dict[cl] = int(xm/len(train_dict[cl]))

## sample cases
random.seed(1)
train_ids = train_ids_all

y_train = [groups[i] for i in train_ids]
y_test = [groups[i] for i in test_ids]

print("total number of cases for building umap:" + str(len(train_ids)))
print(pd.value_counts(y_train))
print("number of events: ~" + str(M))

## estimate how many events that can be sampled
print("Events to be sampled from each case:")
print(xn_dict)

print("selected panel:")
print(sel_panel)

x0 = return_FCS_data(panel=sel_panel, case=good_data[0], data_directory=data_directory, curated=curated_panel)
sel_rows = []
U = np.asarray(x0)

RND_SEED = 1

t1 = datetime.datetime.now()
t1p = time.process_time()
print("TSTAMP A : " + str(t1))

for j in range(len(train_ids)):

    if(j % 100 == 0):
        print("PROGRESS : " + str(j * 100/len(train_ids)))

    i = train_ids[j]

    x0 = return_FCS_data(panel=sel_panel, case=good_data[i], data_directory=data_directory, curated=curated_panel)
    u0 = np.asarray(x0)

    m_i = xn_dict[good_data[i]['cohort']]

    random.seed(RND_SEED)
    ux_i = random.sample(range(u0.shape[0]), min(m_i, u0.shape[0]))
    u0 = u0[ux_i, ]

    sel_rows.append(u0.shape[0])

    if j == 0:
        U = u0
    else:
        U = np.concatenate((U, u0))

if not os.path.exists(save_dir):
    print("creating directory " + save_dir)
    os.makedirs(save_dir)

with open(save_dir + RUN_ID + "_run_vars.obj", "wb") as f:
    pickle.dump([RUN_ID, RND_SEED, PANEL_NUM, M, xn_dict, train_ids, test_ids, y_train, y_test, sel_rows, sel_panel, curated_panel], f)

np.save(save_dir + RUN_ID + "_X.npy", U)

t2 = datetime.datetime.now()
t2p = time.process_time()
print("TSTAMP B : " + str(t2))

print("Process time: B-A = ", t2p-t1p)

ndim = len(curated_panel)
n_components = 2

print("Building umap : " + str(ndim) + " markers -> " + str(n_components) + " dims")

dims = (ndim,1)

print("matrix to build umap: ")
print(U.shape)

embedder = umap.UMAP()
embedding = embedder.fit(U)

t3 = datetime.datetime.now()
t3p = time.process_time()
print("TSTAMP C : " + str(t3))

print("time elapsed: " + str(t3-t2))
print("Process time: C-B = ", t3p-t2p)

print("Done. Saving...")

np.save(save_dir + RUN_ID + "_X_embedding.npy", embedder.embedding_)

with open(save_dir + RUN_ID + "_umap.obj", "wb") as f:
    pickle.dump(embedder, f)

t4 = datetime.datetime.now()
t4p = time.process_time()
print("TSTAMP D : " + str(t4))

print("time elapsed: " + str(t4-t3))
print("Process time: D-C = ", t4p-t3p)

print("done.")
