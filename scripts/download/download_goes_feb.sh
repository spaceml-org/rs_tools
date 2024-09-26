#!/bin/bash

set -o xtrace

SAVE_DIR="/mnt/disks/goes-data/"

START_DATE="2020-02-01" # repeat download for cloudmask!!!
END_DATE="2020-02-28"
TIME_STEP="01:00:00"

# python rs_tools/_src/data/msg/downloader_msg_modis_overpass.py --save-dir $SAVE_DIR --start-date $START_DATE --end-date $END_DATE
python rs_tools/_src/data/goes/downloader_goes16.py --save-dir $SAVE_DIR --start-date $START_DATE --end-date $END_DATE --time-step $TIME_STEP

