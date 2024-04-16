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
import multiprocessing
from multiprocessing import Pool, cpu_count
import tensorflow as tf
from tensorflow.keras import layers
import umap
from umap.parametric_umap import load_ParametricUMAP

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
        self.log = open("4_assemble_"+ RUN_ID + ".logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()

print("============ RUN_ID:" + RUN_ID + "============")

flist = os.listdir("../out/" + RUN_ID + "_histograms/")

flist.sort()

print("file list length: " + str(len(flist)))

t1 = datetime.datetime.now()
print("TSTAMP A : " + str(t1))


x_list = []
for i in range(len(flist)):
    if i % 1000 == 1:
        print(str(i) + "/" + str(len(flist)) + " loaded (t = " + str(datetime.datetime.now()) + ")" )
    fi = flist[i]
    xi = np.load("../out/" + RUN_ID + "_histograms/" + fi)
    x_list.append(xi.reshape(xi.shape[0] * xi.shape[1], 1))


H = np.concatenate((x_list), axis=1)

print(H.shape)

np.save("../out/" + RUN_ID + "_H.npy", H)

t2 = datetime.datetime.now()
print("TSTAMP B : " + str(t2))

print("time elapsed: " + str(t2-t1))

print("done.")
