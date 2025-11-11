#!/bin/bash

set -o xtrace

# INPUT_DIR="/mnt/disks/eo-miniset/geoprocessed/msg"
# SAVE_DIR="/home/anna.jungbluth/metrics/eo-miniset/msg"

# python rs_tools/_src/preprocessing/compile_metrics.py --save_dir $SAVE_DIR --input_dir $INPUT_DIR

# INPUT_DIR="/mnt/disks/eo-miniset/geoprocessed/goes"
# SAVE_DIR="/home/anna.jungbluth/metrics/eo-miniset/goes"


# INPUT_DIR="/home/annajungbluth/converted/goes-3000m/"
# SAVE_DIR="/home/annajungbluth/metrics/goes-3000m/"

INPUT_DIR="/home/annajungbluth/converted/msg/"
SAVE_DIR="/home/annajungbluth/metrics/msg/"

python rs_tools/_src/preprocessing/compile_metrics.py --save_dir $SAVE_DIR --input_dir $INPUT_DIR
