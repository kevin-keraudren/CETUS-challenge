#!/usr/bin/python

import irtk
import os
import sys
from glob import glob
import subprocess

patient_id = sys.argv[1]

remote = "/run/user/kevin/gvfs/sftp:host=biomedia07.doc.ic.ac.uk,user=kpk09"

if not os.path.exists("predictions/"+patient_id+"/movie"):
    os.makedirs("predictions/"+patient_id+"/movie")

all_frames = sorted(glob(remote+"/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Testing/nifti/"+patient_id+"_frame*.nii.gz"))
for f in all_frames:
    if "_seg" in f:
        continue
    print f

    name = os.path.basename(f)[:-len(".nii.gz")]
    print name

    for i in range(1,5):
        cmd = [ "display",
                f,
                "predictions/"+patient_id+"/iter"+str(i)+"_"+name+"_hard.nii.gz",
                "-x","1024",
                "-y","768",
                "-res","2",
                "-cursor",
                "-scontour", "255 0 0",
                "-line", str(2),
                "-offscreen","predictions/"+patient_id+"/movie/iter"+str(i)+"_"+ name + ".png" ]
        
        print " ".join(cmd)
        proc = subprocess.Popen( cmd )
        (out, err) = proc.communicate()
        if out is not None:
            print out
        if err is not None:
            print err
