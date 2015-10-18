#!/usr/bin/python

import irtk
from glob import glob
import os
import sys
import numpy as np

if not os.path.exists("png"):
    os.makedirs("png")

if len(sys.argv) == 1:
    all_files = glob( "predictions/*/iter4_*_hard.nii.gz" )
else:
    all_files = glob( "predictions/"+sys.argv[1]+"/iter4_*_hard.nii.gz" )

for f in all_files:
    print f
    name = os.path.basename(f)[len("iter4_"):-len("_hard.nii.gz")]
    img = irtk.imread("denoised/"+name+".nii.gz",dtype='int32')
    mask = irtk.imread(f).transform(target=img.get_header(),interpolation="nearest")
    irtk.imshow(img,mask,filename="png/"+name+".png",opacity=0.4)
