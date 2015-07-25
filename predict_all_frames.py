#!/usr/bin/python

import irtk
import heartdetector
import numpy as np
from glob import glob

import sys
import os
import argparse
from time import time

def get_ED_ES(patient_id):
    f = "/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Testing/Images/"+patient_id+"/"+patient_id+"_ED_ES_time.txt"
    f = open(f,"rb")
    res = []
    for line in f:
        line = line.rstrip() # chomp
        res.append( int(line.split(' ')[-1]) - 1 )
    return res

parser = argparse.ArgumentParser(
    description="""Left ventricule segmentation using Autocontext Random Forests.
Code for the MICCAI 2014 Segmentation challenge.""" )
parser.add_argument( 'patient_id', type=str )
parser.add_argument( '--forest', type=str, required=True )
parser.add_argument( '--frame', type=int, default=None )
parser.add_argument( '--all', action="store_true", default=False )
parser.add_argument( '--time', action="store_true", default=False )
parser.add_argument( '--nb_autocontext', type=int, default=4 )
parser.add_argument( '--debug', action="store_true", default=False )

args = parser.parse_args()

if not args.time:
    print args
    
start = time()
    
detector = heartdetector.HeartDetector( name=args.forest )
detector.load()

if not args.time:
    print detector

if not os.path.exists("predictions/"+args.patient_id):
    os.makedirs("predictions/"+args.patient_id)

all_frames = sorted(glob("/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Testing/standard_input_data/"+args.patient_id+"_frame*.nii.gz"))

tmp = irtk.imread(all_frames[0])
mask = irtk.zeros(tmp.get_header(),dtype='float32')
for f in all_frames:
    if "_seg" in f:
        continue
    mask += irtk.imread(f)

mask = (mask > 0).astype('uint8')

if args.frame is not None:
    all_frames = [all_frames[args.frame]]
elif not args.all:
    ED,ES = get_ED_ES(args.patient_id)
    all_frames = [all_frames[ED],all_frames[ES]]
    
for f in all_frames:
    if "_seg" in f:
        continue
    if not args.time:
        print f
  
    all_proba = heartdetector.predict( detector,
                                       f,
                                       ga=0.0,
                                       nb_autocontext=args.nb_autocontext,
                                       mask=mask,
                                       debug=args.debug,
                                       return_all=not args.time )
    if isinstance(all_proba,irtk.Image):
        irtk.imwrite("predictions/"+args.patient_id+"/iter"+str(args.nb_autocontext)+"_"+os.path.basename(f),all_proba)
        irtk.imwrite("predictions/"+args.patient_id+"/iter"+str(args.nb_autocontext)+"_"+os.path.basename(f)[:-len('.nii.gz')]
                     +"_hard.nii.gz",
                     detector.groups[0].hard_thresholding( all_proba[1].resample(0.0005),
                                                           smoothing=4.0*0.001 ) )

    else:        
        for i,proba in enumerate(all_proba,start=1):
            irtk.imwrite("predictions/"+args.patient_id+"/iter"+str(i)+"_"+os.path.basename(f),proba)
            irtk.imwrite("predictions/"+args.patient_id+"/iter"+str(i)+"_"+os.path.basename(f)[:-len('.nii.gz')]
                     +"_hard.nii.gz",
                     detector.groups[0].hard_thresholding( proba[1].resample(0.0005),
                                                           smoothing=4.0*0.001 ) )


stop = time()
print stop - start

