#!/usr/bin/python

import irtk
from glob import glob
import os

def get_ED_ES(patient_id):
    f = "/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Training/Images/"+patient_id+"/"+patient_id+"_ED_ES_time.txt"
    f = open(f,"rb")
    res = []
    for line in f:
        line = line.rstrip() # chomp
        res.append( int(line.split(' ')[-1]))
    return res

all_files = glob("nifti/*_seg.nii.gz")

m = 0.0
n = 0.0
for f in all_files:
    patient_id = os.path.basename(f).split("_")[0]
    frame_id = int(os.path.basename(f).split("_")[1][len("frame"):])
    ED_ES = get_ED_ES(patient_id)
    print patient_id, frame_id, ED_ES
    if frame_id == ED_ES[1]:
        mask = irtk.imread(f).resample(0.001, interpolation='nearest')
        m += mask.sum()
        n += 1

m /= n

print "mean heart volume:",m
