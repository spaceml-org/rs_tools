#!/bin/bash

set -o xtrace

INPUT_DIR="/mnt/disks/eo-data/msg"
SAVE_DIR="/home/anna.jungbluth/metrics/msg"

python rs_tools/_src/preprocessing/check_quality.py --save_dir $SAVE_DIR --input_dir $INPUT_DIR

INPUT_DIR="/mnt/disks/eo-data/goes"
SAVE_DIR="/home/anna.jungbluth/metrics/goes"

python rs_tools/_src/preprocessing/check_quality.py --save_dir $SAVE_DIR --input_dir $INPUT_DIR
