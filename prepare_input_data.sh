#!/bin/bash

set -x
set -e

python prepare_input_data.py
./data_augmentation.sh
