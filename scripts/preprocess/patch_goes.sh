#!/bin/bash

set -o xtrace

PATCH_SIZE="450"
NUM_PATCHES="100"
READ_PATH="/mnt/disks/eo-miniset/geoprocessed/goes"
SAVE_PATH="/home/anna.jungbluth/tmp-data/goes"

python rs_tools/_src/preprocessing/centerpatcher.py --read-path $READ_PATH --save-path $SAVE_PATH --patch-size $PATCH_SIZE --num-patches $NUM_PATCHES