#!/bin/bash

set -x
set -e

output_folder=/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Training/nifti

for f in Images/Patient*/*.mhd
do
    python `which convert.py` $f $output_folder/`basename $f .mhd`.nii.gz
done

for f in Truth/Patient*/*_ED_truth.vtk
do
    name=`basename $f .vtk`
    patient_id=`perl -e '@list = split /_/, $ARGV[0]; print $list[0]' $name`
    frame_id=`sed -n '1p' Images/${patient_id}/${patient_id}_ED_ES_time.txt | perl -ne 'chomp; @a = split " "; printf "%02d",$a[3]'`
    python `which polydata2nifti.py` $f Images/${patient_id}/${patient_id}_frame${frame_id}.mhd $output_folder/${patient_id}_frame${frame_id}_seg.nii.gz
done

for f in Truth/Patient*/*_ES_truth.vtk
do
    name=`basename $f .vtk`
    patient_id=`perl -e '@list = split /_/, $ARGV[0]; print $list[0]' $name`
    frame_id=`sed -n '2p' Images/${patient_id}/${patient_id}_ED_ES_time.txt | perl -ne 'chomp; @a = split " "; printf "%02d",$a[3]'`
    python `which polydata2nifti.py` $f Images/${patient_id}/${patient_id}_frame${frame_id}.mhd $output_folder/${patient_id}_frame${frame_id}_seg.nii.gz
done
