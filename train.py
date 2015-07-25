#!/usr/bin/python

import heartdetector
import numpy as np

params = { 'img_folder':"/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Training/input_data",
           'seg_folder':"/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Training/input_data",
           'ga_files':[],
           'name':"forest_border_metadata_highres",
           'nb_samples':2000,
           'nb_background_samples':2000,
           'nb_tests':1000,
           'verbose':True,
           'n_jobs':2,
           'parallel':-1,
           'test':"heartautocontext",
           #'test':"autocontextGradient",
           #'test':"autocontextDistancePrior",
           #'use_extra_layers':["slic_entropy","gradient","maximum","minimum"],
           #'use_extra_layers':["dx","dy","dz"],
           'use_extra_layers':[],
           'use_background_distance':False,
           'use_world_align':True,
           'resample':np.array([0.00065,0.00065,0.00065,1],dtype='float32'),
           'ksampling':1.0,
           'dx':100,
           'dy':100,
           'dz':100,
           'cx':100,
           'cy':100,
           'cz':100,
           'n_estimators' : 30,
           'max_depth' : 30           
           }
detector = heartdetector.HeartDetector( **params )
print detector

all_patients = sorted(heartdetector.get_patients(detector.params['seg_folder'])[::4])
#print all_patients,len(all_patients)

# n_testing = 5
# training_patients = all_patients[:-n_testing]
# testing_patients = all_patients[-n_testing:]

#training_patients, testing_patients = heartdetector.split_patients(all_patients,5)
    
#print training_patients
detector.fit( all_patients,
              max_autocontext=10,
              min_dice_score=0.99,
              n_validation=15 )

print detector.info

print "will predict testing data..."
testing_score = detector.score(all_patients,nb_autocontext=len(detector.info['improvements'])-1)
print testing_score
detector.email_log("Testing score: " + str(testing_score))
