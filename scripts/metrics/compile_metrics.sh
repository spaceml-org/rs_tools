#!/bin/bash

set -o xtrace

# INPUT_DIR="/mnt/disks/eo-miniset/geoprocessed/msg"
# SAVE_DIR="/home/anna.jungbluth/metrics/eo-miniset/msg"

# python rs_tools/_src/preprocessing/compile_metrics.py --save_dir $SAVE_DIR --input_dir $INPUT_DIR

INPUT_DIR="/mnt/disks/eo-miniset/geoprocessed/goes"
SAVE_DIR="/home/anna.jungbluth/metrics/eo-miniset/goes"

python rs_tools/_src/preprocessing/compile_metrics.py --save_dir $SAVE_DIR --input_dir $INPUT_DIR
