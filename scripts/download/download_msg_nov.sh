#!/bin/bash

set -o xtrace

SAVE_DIR="/mnt/disks/data/msg/2010/"

START_DATE="2010-11-01" # repeat download for cloudmask!!!
END_DATE="2010-11-30"

# python rs_tools/_src/data/msg/downloader_msg_modis_overpass.py --save-dir $SAVE_DIR --start-date $START_DATE --end-date $END_DATE
python rs_tools/_src/data/msg/downloader_msg.py --save-dir $SAVE_DIR --start-date $START_DATE --end-date $END_DATE

