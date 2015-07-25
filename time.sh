#!/bin/bash

set -x
set -e

for i in {16..30}
do
    patient_id=Patient${i}
    python predict_all_frames.py $patient_id 0 --time --nb_autocontext 4
done
