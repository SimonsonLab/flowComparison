#!/bin/sh
set -e
set -u

python3 1_make_2dhists.py 50

python3 2_run_EnsembleCNN.py 50
