#!/usr/bin/python

import irtk
import numpy as np

from glob import glob

import sys
import os

import joblib
from joblib import Parallel, delayed

from skimage import morphology
from skimage.morphology import watershed
import scipy.ndimage as nd

from scipy.stats.mstats import mquantiles

from organdetector import OrganDetector

def preprocess_training_data( patient_id,
                              img_folder,
                              seg_folder,
                              resample,
                              offline=False,
                              online=True):
    if offline or online:
        if ( offline
             and os.path.exists( "offline_preprocessing/"+patient_id+"_img.nii.gz" )
             and os.path.exists( "offline_preprocessing/"+patient_id+"_seg.nii.gz" ) ):
                 return
        img = irtk.imread( img_folder + "/" + patient_id + ".nii.gz",
                           dtype='float32' )
        seg = irtk.imread( seg_folder +"/"+patient_id+"_seg.nii.gz",
                           dtype="uint8" )

        wall = nd.binary_dilation( seg,
                                   morphology.ball(int(12.5*0.001/seg.header['pixelSize'][0])) )
        wall = wall.astype('int')
        points = np.transpose(np.nonzero(wall))[::4]
        center,S,V = fit_ellipsoidPCA( points )
        if V[0,0] < 0:
            V *= -1
        
        points = np.transpose(np.nonzero(wall))
        projections = np.dot(points-center,V[0])

        # valves
        index = projections > (projections.max() - 40.0*0.001/seg.header['pixelSize'][0])

        #print "VALVE size:",np.sum(index), projections.max(), 40.0*0.001/seg.header['pixelSize'][0]
    
        wall[points[index,0],
             points[index,1],
             points[index,2]] = 2

        #print "VALVE1", wall.max()

        wall = irtk.Image(wall,seg.get_header())
    
        img = img.resample( pixelSize=resample, interpolation='linear' ).rescale(0,1000)
        seg = seg.transform(target=img,interpolation="nearest").astype('uint8')
        wall = wall.transform(target=img,interpolation="nearest").astype('uint8')
 
        wall[seg>0] = 0
        seg[wall==1] = 2
        seg[wall==2] = 3

        #print "VALVE2", seg.max()
    
        #irtk.imwrite("debug/"+patient_id+"_border.nii.gz",seg)
    
        seg[img==0] = 255

        if offline:
            irtk.imwrite( "offline_preprocessing/"+patient_id+"_img.nii.gz", img )
            irtk.imwrite( "offline_preprocessing/"+patient_id+"_seg.nii.gz", seg )
            return

    if not online:
        img = irtk.imread( "offline_preprocessing/"+patient_id+"_img.nii.gz" )
        seg = irtk.imread( "offline_preprocessing/"+patient_id+"_seg.nii.gz" )
        
    mask = irtk.ones( img.get_header(), dtype='uint8' )
    mask[img==0] = 0

    return { 'patient_id': patient_id,
             'img' : img,
             'seg' : seg,
             'mask' : mask }

def split_patients(files,n):
    training_patients = ["Patient"+str(i)+"_" for i in range(1,15-n+1)]
    testing_patients = ["Patient"+str(i)+"_" for i in range(15-n,15+1)]
    training_files = []
    testing_files = []
    for f in files:
        testing = False
        for p in testing_patients:
            if p in os.path.basename(f):
              testing_files.append(f)
              testing = True
              break
        if not testing:
            training_files.append(f)

    return training_files,testing_files

def fit_ellipsoidPCA( points,
                   factor=1.96,
                   spacing=[1.0,1.0,1.0] ):
    """
    1.96 in order to contain 95% of the data
    http://en.wikipedia.org/wiki/1.96
    points are ZYX, spacing is XYZ
    """
    spacing = np.array(spacing[::-1],dtype='float')
    points = points.astype('float')*spacing

    center = points.mean(axis=0)
    points -= center
    # The singular values are sorted in descending order.
    U, S, V = np.linalg.svd(points, full_matrices=False)

    S *= factor
    S /= np.sqrt(len(points)-1)
    
    return center,S,V

def get_patients(seg_folder):
    seg_files = glob(seg_folder+"/*_seg.nii.gz")
    patients = []
    for f in seg_files:
        patient_id = os.path.basename(f)[:-len('_seg.nii.gz')]
        patients.append(patient_id)
    return patients

if __name__ == "__main__":
    params = { 'img_folder':"/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Training/input_data",
               'seg_folder':"/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Training/input_data",
               'name':"debug_forest",
               'nb_samples':500,
               'nb_background_samples':500,
               'nb_tests':500,
               'verbose':True,
               'n_jobs':5,
               'parallel':-1,
               'test':"autocontext",
               'resample':np.array([0.001,0.001,0.001,1],dtype='float32'),
               'ksampling':1.0,
               'dx':30,
               'dy':30,
               'dz':30,
               'cx':30,
               'cy':30,
               'cz':30,
               'lambda_gdt' : 100,
               'labels' : [ "left_ventricule",
                            "wall",
                            "valves" ],
               'preprocessing_function' : preprocess_training_data }
    detector = OrganDetector( **params )
    print detector
    
    all_patients = get_patients(detector.params['seg_folder'])
    np.random.shuffle( all_patients )
    
    n_testing = 5
    n_training = 5

    detector.fit( all_patients[:n_training],
                  max_autocontext=10,
                  min_dice_score=0.99,
                  n_validation=10 )

    print detector.info

    print "will predict testing data..."
    testing_score = detector.score(all_patients[-n_testing:],nb_autocontext=len(detector.info['improvements'])-1)
    print testing_score
    detector.email_log("Testing score: " + str(testing_score))
