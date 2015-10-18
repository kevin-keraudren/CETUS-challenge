#!/bin/bash

set -x
set -e

output_folder=nifti
output_folder_resampled=nifti_resampled
output_folder_denoised=denoised

mkdir -p $output_folder
mkdir -p $output_folder_resampled
mkdir -p $output_folder_denoised

start=1
end=45
for i in $(seq $start $end)
do
    patient_id=Patient${i}
    f=Images/${patient_id}/${patient_id}_ED_ES_time.txt

    # ED
    {
        frame_id=`sed -n '1p' $f | perl -ne 'chomp; @a = split " "; printf "%02d",$a[3]'`
        python `which convert.py` Images/${patient_id}/${patient_id}_frame${frame_id}.mhd $output_folder/${patient_id}_frame${frame_id}.nii.gz

        resample $output_folder/${patient_id}_frame${frame_id}.nii.gz  $output_folder_resampled/${patient_id}_frame${frame_id}.nii.gz  -size 0.001 0.001 0.001 -bspline

		python denoise.py $output_folder_resampled/${patient_id}_frame${frame_id}.nii.gz $output_folder_denoised/${patient_id}_frame${frame_id}.nii.gz
    } &

    # ES
    {
        frame_id=`sed -n '2p' $f | perl -ne 'chomp; @a = split " "; printf "%02d",$a[3]'`
        python `which convert.py` Images/${patient_id}/${patient_id}_frame${frame_id}.mhd $output_folder/${patient_id}_frame${frame_id}.nii.gz

        resample $output_folder/${patient_id}_frame${frame_id}.nii.gz  $output_folder_resampled/${patient_id}_frame${frame_id}.nii.gz  -size 0.001 0.001 0.001 -bspline

		python denoise.py $output_folder_resampled/${patient_id}_frame${frame_id}.nii.gz $output_folder_denoised/${patient_id}_frame${frame_id}.nii.gz
    }

done

for f in Truth/Patient*_ED_truth.vtk
do
    name=`basename $f .vtk`
    patient_id=`perl -e '@list = split /_/, $ARGV[0]; print $list[0]' $name`
    frame_id=`sed -n '1p' Images/${patient_id}/${patient_id}_ED_ES_time.txt | perl -ne 'chomp; @a = split " "; printf "%02d",$a[3]'`
	python `which polydata2nifti.py` $f Images/${patient_id}/${patient_id}_frame${frame_id}.mhd $output_folder/${patient_id}_frame${frame_id}_seg.nii.gz
    python `which polydata2nifti.py` $f $output_folder_resampled/${patient_id}_frame${frame_id}.nii.gz $output_folder_resampled/${patient_id}_frame${frame_id}_seg.nii.gz
done

for f in Truth/Patient*_ES_truth.vtk
do
    name=`basename $f .vtk`
    patient_id=`perl -e '@list = split /_/, $ARGV[0]; print $list[0]' $name`
    frame_id=`sed -n '2p' Images/${patient_id}/${patient_id}_ED_ES_time.txt | perl -ne 'chomp; @a = split " "; printf "%02d",$a[3]'`
	python `which polydata2nifti.py` $f Images/${patient_id}/${patient_id}_frame${frame_id}.mhd $output_folder/${patient_id}_frame${frame_id}_seg.nii.gz
    python `which polydata2nifti.py` $f $output_folder_resampled/${patient_id}_frame${frame_id}.nii.gz $output_folder_resampled/${patient_id}_frame${frame_id}_seg.nii.gz
done
