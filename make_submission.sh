#!/bin/bash

set -x
set -e

iter=$1

output_folder=submission

mkdir -p $output_folder

for i in {31..45}
do
    patient_id=Patient${i}
    frame_id_ED=`sed -n '1p' Images/${patient_id}/${patient_id}_ED_ES_time.txt | perl -ne 'chomp; @a = split " "; printf "%02d",$a[3]'`
    frame_id_ES=`sed -n '2p' Images/${patient_id}/${patient_id}_ED_ES_time.txt | perl -ne 'chomp; @a = split " "; printf "%02d",$a[3]'`
    mcubes predictions/${patient_id}/iter${iter}_${patient_id}_frame${frame_id_ED}_hard.nii.gz ${output_folder}/${patient_id}_ED_result.vtk 1 -smooth 1000 &
    mcubes predictions/${patient_id}/iter${iter}_${patient_id}_frame${frame_id_ES}_hard.nii.gz ${output_folder}/${patient_id}_ES_result.vtk 1 -smooth 1000 &
done

