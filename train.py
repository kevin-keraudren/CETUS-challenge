#!/usr/bin/python

import heartdetector
import numpy as np
import pprint


params = { 'img_folder':"/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Training/standard_input_data",
           'seg_folder':"/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Training/standard_input_data",
           'name':"forest_autocontextN315",
           'nb_samples':500,
           'nb_background_samples':500,
           'nb_tests':500,
           'verbose':True,
           'n_jobs':5,
           'parallel':-1,
           'test':"autocontextN",
           'resample':np.array([0.001,0.001,0.001,1],dtype='float32'),
           'ksampling':1.0,
           'dx':30,
           'dy':30,
           'dz':30,
           'cx':30,
           'cy':30,
           'cz':30,
           'lambda_gdt' : 100
           }

detector = heartdetector.HeartDetector( **params )
print detector

all_patients = heartdetector.get_patients(detector.params['seg_folder'])

# print "doing offline preprocessing..."
# detector.do_offline_preprocessing( all_patients, 30 )
# exit(0)

for i in xrange(5):
    print "learning autocontext", i
    np.random.shuffle( all_patients )
    detector.fit( all_patients[:315],
                  start=i,
                  max_autocontext=i+1,
                  min_dice_score=0.99,
                  n_validation=15 )

pp = pprint.PrettyPrinter(indent=4)
pp.pprint( detector.info )

print "will predict testing data..."
testing_score = detector.score(all_patients[-10:],nb_autocontext=len(detector.info['improvements'])-1)
print testing_score
detector.email_log("Testing score: " + str(testing_score))
