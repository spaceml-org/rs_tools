#!/bin/bash

set -o xtrace

INPUT_DIR="/mnt/disks/msg-data/msg"
SAVE_DIR="/home/anna.jungbluth/metrics/msg"

python rs_tools/_src/preprocessing/compile_metrics.py --save_dir $SAVE_DIR --input_dir $INPUT_DIR

INPUT_DIR="/mnt/disks/goes-data/goes"
SAVE_DIR="/home/anna.jungbluth/metrics/goes"

python rs_tools/_src/preprocessing/compile_metrics.py --save_dir $SAVE_DIR --input_dir $INPUT_DIR
