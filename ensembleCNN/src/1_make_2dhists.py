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

import statistics
import pickle
import os
import logging

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        logfilename = "./log/1_bin" + sys.argv[1] + ".logfile.log"
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

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def return_panel_path(panel, case):
    """
    This function finds the file path for FCS file for the panel for the case.
    Test: return_panel_path(the_marker_panels[0], good_data[0])
    """
    for file in case['filepaths']:
        if panel == file['fcs']['markers']:
            #print(file['fcs']['path'])
            return file['fcs']['path']

def return_FCS_data(panel, case, data_directory):
    """
    Return the FCS file data corresponding the given antibody panel for the given case.
    Test: return_FCS_data(the_marker_panels[0], good_data[0])
    """
    relative_path = return_panel_path(panel, case)
    complete_path = data_directory + relative_path
    meta, data = fcsparser.parse(complete_path, reformat_meta=True)
    channel_names = list(meta['_channel_names_'])
    data = pd.DataFrame(data)
    data.columns = channel_names
    return data

def create_histograms(case, data_directory, num_bins, the_marker_panels, curated_marker_panels):
    heatmaps_list = []
    for b, panel in enumerate(the_marker_panels):

        #return the FCS file data for the given panel
        pddata = return_FCS_data(panel, case, data_directory)

        for i in range(len(curated_marker_panels[b])):
            for j in range(i+1, len(curated_marker_panels[b])):
                x = pddata[curated_marker_panels[b][i]]/1024
                y = pddata[curated_marker_panels[b][j]]/1024

                #Create the 2d histogram.
                heatmap, xedges, yedges = np.histogram2d(y,x, bins=num_bins, range=[[0,1],[0,1]])
                #Apply log function.  Add 1 to avoid negative values.
                heatmap = np.log(heatmap + 1)
                #Scale 0 to 1
                heatmap = heatmap/np.amax(heatmap)

                #Display the heatmap.
                #plt.clf()
                #plt.imshow(heatmap, origin='lower', cmap=plt.cm.binary)
                #plt.colorbar()
                #plt.show()

                #Add it to the list
                heatmap.flatten()
                heatmaps_list.append(heatmap)
    return np.concatenate(heatmaps_list).flatten()

print("the marker panels:")
print(the_marker_panels)

print("curated marker panels:")
print(curated_marker_panels)

if len(sys.argv) <= 1:
    num_bins = 25
else:
    num_bins = int(sys.argv[1])

print("num_bins=" + str(num_bins))

t1 = datetime.datetime.now()
t1p = time.process_time()
print("TSTAMP A : " + str(t1))

## parallelize and sort
hists = []
for i in range(0, len(good_data)):
    print(i)
    x_out = None
    try:
        x_out = create_histograms(good_data[i], data_directory, num_bins, the_marker_panels, curated_marker_panels)
    except Exception as e:
        pass

    hists.append(x_out)

t2 = datetime.datetime.now()
t2p = time.process_time()
print("TSTAMP B : " + str(t2))

print("time elapsed: " + str(t2-t1))

print("Process time: B-A = ", t2p-t1p)

savefile = save_dir + "X_" + str(num_bins) + ".npy"

print("saving at", savefile)

np.save(savefile, np.stack(hists), allow_pickle=False)

print("done.")
