#!/bin/bash

set -x
set -e

start=$1
end=$2

for i in $(seq $start $end)
do
    patient_id=Patient${i}
    python predict_all_frames.py $patient_id --forest forest_615 \
                                             --nb_autocontext 4 &
done
