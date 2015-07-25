#!/usr/bin/python

from glob import glob
import shutil
import os

denoised = glob("dictionary-denoised/*")

for f in denoised:
    name = os.path.basename(f)[:-len("_denoised.nii.gz")]
    print name
    shutil.copy( f, "input_data/"+name+".nii.gz" )
    if os.path.exists( "nifti/"+name+"_seg.nii.gz"):
        shutil.copy( "nifti/"+name+"_seg.nii.gz", "input_data/"+name+"_seg.nii.gz" )
    else:
        patient_id, frame_id = name.split('_')
        segmentations = glob("deformable_registration/propogated_segmentations/"+patient_id+"_*_to_"+frame_id+"_seg.nii.gz")
        n = int(frame_id[-2:])
        ref1 = int(os.path.basename(segmentations[0]).split('_')[1][-2:])
        ref2 = int(os.path.basename(segmentations[1]).split('_')[1][-2:])
        if abs(n-ref1) < abs(n-ref2):
            shutil.copy( segmentations[0], "input_data/"+name+"_seg.nii.gz" )
        else:
            shutil.copy( segmentations[1], "input_data/"+name+"_seg.nii.gz" )
            
