#!/bin/bash

set -x
set -e

for i in {16..30}
do
    patient_id=Patient${i}
    python predict_all_frames.py $patient_id --forest forest_autocontextN315 \
                                             --nb_autocontext 4 &
done
