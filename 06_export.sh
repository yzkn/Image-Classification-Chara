#!/bin/bash

# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

# remove the restriction of path strength before this pip command (LongPathsEnabled in the regedit)
# $ pip install tensorflowjs

INPUT_H5FILE='dataset/weight/finetuning.h5'
OUTPUT_DIR='tfjs'
tensorflowjs_converter --input_format=keras $INPUT_H5FILE $OUTPUT_DIR