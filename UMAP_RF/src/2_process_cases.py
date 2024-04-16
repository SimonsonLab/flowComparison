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

RUN_ID = sys.argv[1]

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("./log/2_"+ RUN_ID + ".logfile.log", "a")

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

## vars
data_directory = '/prj0133/ResearchData/2020FlowcatData/flowCatData/decCLL-9F/'
if not os.path.exists(data_directory):
    data_directory = '/athena/marchionnilab/scratch/lab_data/wikum/2020FlowcatData/flowCatData/decCLL-9F/'
save_dir = '../out/'

sel_classes = ['LPL', 'MCL', 'PL', 'normal', 'MZL', 'CLL', 'MBL', 'HCL', 'FL'] ## no HCLv

with open('../obj/good_data.json') as f:
    good_data = json.load(f)

print("good data length: " + str(len(good_data)))

with open('../obj/good_data_vars.obj', 'rb') as f:
    num_bins, num_columns, total_histograms, marker_panels, the_marker_panels, curated_marker_panels, data_cohorts, data_diagnoses = pickle.load(f)

with open(save_dir + RUN_ID + "_run_vars.obj", "rb") as f:
    RUN_ID, RND_SEED, PANEL_NUM, M, xn_dict, train_ids, test_ids, y_train, y_test, sel_rows, sel_panel, curated_panel = pickle.load(f)

t1 = datetime.datetime.now()
t1p = time.process_time()
print("TSTAMP A : " + str(t1))

print("Loading UMAP...")

with open(save_dir + RUN_ID + "_umap.obj", "rb") as f:
    embedder = pickle.load(f)

t2 = datetime.datetime.now()
t2p = time.process_time()
print("TSTAMP B : " + str(t2))

print("Process time: B-A = ", t2p-t1p)

print("Projection of remaining cases and making histograms: ")

case_proj_dir = save_dir + RUN_ID + "_projections"
case_hist_dir = save_dir + RUN_ID + "_histograms"

if not os.path.exists(case_proj_dir):
    print("creating directory " + case_proj_dir)
    os.makedirs(case_proj_dir)

if not os.path.exists(case_hist_dir):
    print("creating directory " + case_hist_dir)
    os.makedirs(case_hist_dir)

current_files = os.listdir(case_hist_dir)
# all_files = [u['id'] + '.npy' for u in good_data]
all_files = [u['id'] + '.npy' for u in good_data[0:5]]
remaining_files = list(set(all_files) - set(current_files))
id_file_dict = dict(zip([x['id']+'.npy' for x in good_data], range(len(good_data))))
sel_id = [id_file_dict[xid] for xid in remaining_files]

print(len(sel_id), " cases to be processed")

def get_transformation(case):
    sf1 = case_proj_dir + "/" + str(case['id']) + '.npy'
    sf2 = case_hist_dir + "/" + str(case['id']) + '.npy'

    if not os.path.isfile(sf2):

        x0 = return_FCS_data(panel=sel_panel, case=case, data_directory=data_directory, curated=curated_panel)

        u0 = np.asarray(x0)

        v0 = embedder.transform(u0)
        np.save(sf1, v0)

        w0 = convert_case_to_histo(v0)
        np.save(sf2, w0)

    # w = w0.reshape(w0.shape[0] * w0.shape[1], 1)
    return 1
    #print("processed " + str(case['id']))

w_list = []
with multiprocessing.Pool() as pool:
    w_list = pool.map(get_transformation, [good_data[i] for i in sel_id])

t3 = datetime.datetime.now()
t3p = time.process_time()
print("TSTAMP C : " + str(t3))

print("time elapsed: " + str(t3-t2))
print("Process time: C-B = ", t3p-t2p)

print("done.")
