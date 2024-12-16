#!/bin/bash

set -o xtrace

PATCH_SIZE="150"
NUM_PATCHES="100"
READ_PATH="/mnt/disks/eo-miniset/geoprocessed/msg"
SAVE_PATH="/home/anna.jungbluth/tmp-data/msg"

python rs_tools/_src/preprocessing/centerpatcher.py --read-path $READ_PATH --save-path $SAVE_PATH --patch-size $PATCH_SIZE --num-patches $NUM_PATCHES