#!/bin/bash

set -x
set -e

script=/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Training/data_augmentation.py

cd input_data
for f in Patient*_seg.nii.gz
do
    name=`perl -e '@list = split /_/, $ARGV[0]; print $list[0]."_".$list[1]' $f`
    # python $script ${name}.nii.gz ${name}_seg.nii.gz 1 &
    # python $script ${name}.nii.gz ${name}_seg.nii.gz 2 # &
    python $script ${name}.nii.gz ${name}_seg.nii.gz 3 &
    python $script ${name}.nii.gz ${name}_seg.nii.gz 4
    # python $script ${name}.nii.gz ${name}_seg.nii.gz 5
done

cd ..
