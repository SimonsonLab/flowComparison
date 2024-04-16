#!/bin/sh
set -e
set -u

python3 1_sample_events.py 0

read -r RUNID < RUNID.txt

python3 2_process_cases.py $RUNID
