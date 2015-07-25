#!/usr/bin/python

import irtk
import numpy as np
import pickle

import smtplib
from email.mime.text import MIMEText
import socket

import pprint
from glob import glob

import sys
import os
import gc
sys.path.append("/vol/biomedic/users/kpk09/forest/forest/")
from lib.integralforest import integralForest
from integralforest_aux import *

import joblib
from joblib import Parallel, delayed
import scipy.ndimage as nd
from scipy.stats.mstats import mquantiles
import itertools

def predict_autocontext( self,
                         img,
                         mask,
                         nb_labels,
                         nb_autocontext,
                         debug=False,
                         return_all=False ):
    proba = np.ones((nb_labels,img.shape[0],img.shape[1],img.shape[2]),dtype='float32')
    proba /= nb_labels

    header = img.get_header()
    header['dim'][3] = nb_labels
    proba = irtk.Image(proba,header,squeeze=False)
    
    all_steps = []

    for k in xrange(nb_autocontext):
        knowledge = self.get_knowledge(img,proba,mask=mask)

        if debug:
            irtk.imwrite("knowledge_"+str(k)+".nii.gz", knowledge)

        forest = integralForest( folder=self.folder(k),
                                 test=self.params['test'],
                                 parallel=self.params['parallel'],
                                 nb_knowledge_layers=knowledge.shape[0] )
        proba = forest.predict_autocontext( img,
                                            knowledge,
                                            mask,
                                            self.params['ksampling'] )
        proba = irtk.Image(proba,header,squeeze=False)
        if return_all:
            all_steps.append( proba.copy() )
        if debug:
            irtk.imwrite("debug_"+str(k)+".nii.gz", proba)
        
        if k < 1:
            for i in xrange(proba.shape[0]):
                if i == 0:
                    proba[i] = 0
                else:
                    proba[i] = self.get_center(proba[i])
            
        if debug:
            print "done autocontext", k
        #     irtk.imwrite("debug_rescaled_"+str(k)+".nii.gz", proba)

    if not return_all:
        return proba
    else:
        return all_steps

def predict( self,
             filename,
             nb_autocontext=None,
             mask=None,
             debug=False,
             return_all=False ):
    """
    The prediction function must be defined outside of the main class in order to be used in joblib's Parallel.
    """
    nb_labels = len(self.params['labels'])+1
    
    if nb_autocontext is None:
        nb_autocontext = len(glob(self.params['name'] + "_*"))

    if self.params['predict_preprocessing_function'] is None:
        img = irtk.imread( filename, dtype="float32" )
        img = img.resample( pixelSize=self.params['resample'], interpolation='linear' ).rescale(0,1000)
    else:
        img = self.params['predict_preprocessing_function'](self, filename).copy()
    
    if mask is None:
        mask = irtk.ones( img.get_header(), dtype="uint8" ).as3D()
        mask[img==0] = 0
    else:
        mask = mask.resample( pixelSize=self.params['resample'], interpolation='nearest' ).astype('uint8')
    
    probas = predict_autocontext( self,
                                  img,
                                  mask,
                                  nb_labels,
                                  nb_autocontext,
                                  debug=debug,
                                  return_all=return_all )
    
    return probas

class OrganDetector(object):
    """
    Generic organ localization method using Random Forests in an autocontext framework.

    References:
    -----------

    Pauly, O., Glocker, B., Criminisi, A., Mateus, D., Moller, A., Nekolla, S., Navab,
    N.: Fast Multiple Organ Detection and Localization in Whole-body MR Dixon Sequences.
    In: MICCAI. pp. 239-247. Springer (2011)

    Criminisi, A., Shotton, J., Robertson, D., Konukoglu, E.: Regression Forests for
    Efficient Anatomy Detection and Localization in CT Studies. Medical Computer
    Vision. Recognition Techniques and Applications in Medical Imaging pp. 106-117
    (2011)

    Kontschieder, P., Kohli, P., Shotton, J., Criminisi, A.: GeoF: Geodesic Forests
    for Learning Coupled Predictors. In: Computer Vision and Pattern Recognition
    (CVPR), 2013 IEEE Conference on. pp. 65-72. IEEE (2013)

    Tu, Z.: Auto-Context and its Application to High-level Vision Tasks. In: Computer
    Vision and Pattern Recognition (CVPR). pp. 1-8. IEEE (2008)
    """
    
    def __init__( self,
                  n_estimators=20,
                  bootstrap=0.7,
                  max_depth=20,
                  min_items=20,
                  nb_tests=1000,
                  n_jobs=-1, # joblib
                  parallel=-1, # TBB
                  test="autocontext",
                  cx=30,  cy=30,  cz=30,
                  dx=30,  dy=30,  dz=30,
                  ksampling=2.0,
                  verbose=False,
                  img_folder="",
                  seg_folder="",
                  hull_folder="",
                  name=None,
                  nb_samples=2000,
                  nb_background_samples=2000,
                  resample=np.array([2,2,2,1],dtype='float'),
                  lambda_gdt=100,
                  labels=[],
                  training_preprocessing_function=None,
                  predict_preprocessing_function=None,
                  file_extension=".nii.gz",
                  offline_preprocessing=True ):
        if name is None:
            raise ValueError( "you must give a name to your detector, " +
                              "it wil be used to read/write to disk" )
            
        self.params = { "n_estimators" : n_estimators,
                        "bootstrap" : bootstrap,
                        "verbose" : verbose,
                        "max_depth" : max_depth,
                        "min_items" : min_items,
                        "nb_tests" : nb_tests,
                        "n_jobs" : n_jobs,
                        "parallel" : parallel,
                        "test" : test,
                        "cx" : cx, "cy" : cy, "cz" : cz,
                        "dx" : dx, "dy" : dy, "dz" : dz,
                        "ksampling" : ksampling,
                        "img_folder" : img_folder,
                        "seg_folder" : seg_folder,
                        "name" : name,
                        "nb_samples" : nb_samples,
                        "nb_background_samples" : nb_background_samples,
                        "resample" : resample,
                        "lambda_gdt" : lambda_gdt,
                        "training_preprocessing_function" : training_preprocessing_function,
                        "predict_preprocessing_function" : predict_preprocessing_function,
                        "labels" : labels,
                        "file_extension" : file_extension,
                        "offline_preprocessing" : offline_preprocessing }

        self.info = { 'validation_scores' : [],
                      'improvements' : [],
                      'feature_importance' : [] }
        
    def __reduce__(self):
        """
        Required for pickling/unpickling, which is used for instance
        in joblib Parallel.
        An example implementation can be found in numpy/ma/core.py .
        """        
        return ( _DetectorReduce, (self.params,) )

    def __str__(self):
        pp = pprint.PrettyPrinter(indent=4)
        return pp.pformat(self.params) + "\n" + pp.pformat(self.info)

    def __repr__(self):
        return "OrganDetector"       

    def set_params(self, **params):
        for key, value in six.iteritems(params):
            self.params[key] = value
            
    def get_params(self,deep=False):
        if not deep:
            return self.params
        else:
            return copy.deepcopy(self.params)  
    
    def folder(self,autocontext):
        return self.params['name'] + "_" + str(autocontext)
    
    def filename(self):
        return "info_" + self.params['name'] + '.pk'
    
    def save( self ):
        pickle.dump( [ self.params,
                       self.info ],
                     open(self.filename(),"wb"), protocol=-1 )
        return

    def load( self ):
        self.params, self.info = pickle.load( open(self.filename(),"rb") )
        return

    def hard_thresholding( self, proba, smoothing=None ):
        #print "MAX proba:",proba.max(), self.name
        res = proba > 0.5
        res = irtk.largest_connected_component(res,fill_holes=False)
        if smoothing is None:
            return res
        else:
            return res.gaussianBlurring( sigma=smoothing ) >= 0.5

    def soft_thresholding( self, proba ):
        res = proba > 0.5
        res = irtk.largest_connected_component(res,fill_holes=False)
        proba[res==0] = 0
        return proba
    
    def get_center( self, proba ):
        if len(proba.shape) != 3:
            proba = proba.as3D()
        res = irtk.zeros( proba.get_header() ).as3D()
        tmp = proba.view(np.ndarray).copy()
        tmp[self.hard_thresholding(proba)==0] = 0
        if tmp.sum() == 0:
            return res
        center = np.array( nd.center_of_mass( tmp ), dtype='int' )
        res[center[0],center[1],center[2]] = 1
        return res

    def show_offline_preprocessing( self, folder ):
        if not os.path.exists(folder):
            os.makedirs(folder)
        all_files = glob( "offline_preprocessing/*_img.nii.gz" )
        for f in all_files:
            print f
            name = os.path.basename(f)[:-len("_img.nii.gz")]
            img = irtk.imread(f,dtype='float32')
            seg = irtk.imread("offline_preprocessing/"+name+"_seg.nii.gz",dtype='uint8')
            irtk.imshow(img,seg,filename=folder+"/"+name+".png",opacity=0.4)


    def feature_mapping( self ):
        if self.params['test'] == "autocontext":
            return ["Image"] + ["Proba "+l for l in self.params['labels']] + ["Distance "+l for l in self.params['labels']]
        if self.params['test'] == "autocontext2" or self.params['test'] == "autocontextN":
            tmp = ["Proba "+l for l in self.params['labels']] + ["Distance "+l for l in self.params['labels']]
            tmp = [ x+" / "+y for x,y in itertools.product(tmp, repeat=2) ]
            return ["Image"] + tmp
        else:
            raise ValueError("Unknown test")
    
    def get_knowledge( self, img, proba, mask=None ):
        knowledge = proba[1:].rescale(0,1000).view(np.ndarray)
        if len(knowledge.shape) == 2:
            knowledge = knowledge[np.newaxis,np.newaxis,...]
        if len(knowledge.shape) == 3:
            knowledge = knowledge[np.newaxis,...]

        group_distances = Parallel(n_jobs=self.params['n_jobs'])(delayed(gdt)( img,
                                                                               self.get_center(proba[group_id]),
                                                                               l=float(self.params['lambda_gdt'])*self.params['resample'][0] )
                                                                 for group_id in range(1,proba.shape[0]) )

        for group_id in xrange(1,proba.shape[0]):
            knowledge = np.concatenate((knowledge,
                                        np.reshape(group_distances[group_id-1].rescale(0,1000),
                                                   (1,
                                                    img.shape[0],
                                                    img.shape[1],
                                                    img.shape[2]))),
                                       axis=0)

        header = img.get_header()
        header['dim'][3] = knowledge.shape[0]
        knowledge = irtk.Image(knowledge.copy(order='C').astype('float32'),header,squeeze=False)
        if self.params['ksampling'] != 1.0:
            knowledge = knowledge.resample(img.header['pixelSize'][0]*self.params['ksampling'])

        # irtk.imwrite("debug.nii.gz",knowledge)
        # exit(0)

        return knowledge    

    def do_offline_preprocessing( self, patient_ids, n_jobs ):
        Parallel(n_jobs=n_jobs)(delayed(self.params['training_preprocessing_function'])( patient_id,
                                                                   self.params['img_folder'],
                                                                   self.params['seg_folder'],
                                                                   self.params["resample"],
                                                                   offline=True )
                                for patient_id in patient_ids )
        return
        
    def fit( self,
             patient_ids,
             n_validation=5,
             min_dice_score=0.7,
             max_autocontext=10,
             start=0 ):
        """
        Train the classifier.
        """
        nb_labels = len(self.params['labels'])+1
        ## Preprocess data only once to speed up training
        ## (requires more memory)

        # split patients to get validation set
        np.random.shuffle(patient_ids)

        n_validation = min(len(patient_ids)/2,n_validation)
        
        training_patients = patient_ids[:-n_validation]
        validation_patients = patient_ids[-n_validation:]

        self.info['training_patients'] = training_patients
        self.info['validation_patients'] = validation_patients

        if self.params['verbose']:
            print "fitting with", len(training_patients), "training patients and", \
                len(validation_patients), "validation patients"

            print "doing preprocessing..."
        gc.collect()
        training_data = Parallel(n_jobs=self.params['n_jobs'])(delayed(self.params['training_preprocessing_function'])( patient_id,
                                                                                                  self.params['img_folder'],
                                                                                                  self.params['seg_folder'],
                                                                                                  self.params["resample"],
                                                                                                  online=False )
                                                               for patient_id in training_patients )

        if self.params['verbose']:
            print "learning"
            
        i = start
        dice_score = 0
        previous_dice_score = -1
        while ( i < max_autocontext and
                dice_score < min_dice_score and
                dice_score > previous_dice_score ):
            forest = integralForest( ntrees=self.params['n_estimators'],
                                     bagging=self.params['bootstrap'],
                                     max_depth=self.params['max_depth'],
                                     min_items=self.params['min_items'],
                                     nb_tests=self.params['nb_tests'],
                                     parallel=self.params['parallel'],
                                     test=self.params['test'],
                                     cx=self.params['cx'],  cy=self.params['cy'],  cz=self.params['cz'],
                                     dx=self.params['dx'],  dy=self.params['dy'],  dz=self.params['dz'],
                                     nb_labels=nb_labels,
                                     nb_knowledge_layers=2*(nb_labels-1),
                                     ksampling=self.params['ksampling'],
                                     verbose=False,
                                     nfeatures=len(self.feature_mapping())
                                     )

            print "predicting training data"
            tmp_probas = Parallel(n_jobs=self.params['n_jobs'])(delayed(predict_autocontext)( self,
                                                                                              data['img'],
                                                                                              data['mask'],
                                                                                              nb_labels,
                                                                                              i )
                                                                for data in training_data )

            # tmp_probas = []
            # for data in training_data:
            #     tmp_probas.append( predict_autocontext( self,
            #                                             data['img'],
            #                                             data['mask'],
            #                                             data['extra_layers'],
            #                                             data['metadata'],
            #                                             nb_labels,
            #                                             all_ga[data['patient_id']],
            #                                             i ) )
            
            for data,proba in zip(training_data,tmp_probas):
                img = data['img']
                mask = data['mask']
                seg = data['seg'].copy()

                # kind of bootstrapping
                for l in range(proba.shape[0]):
                    correct = np.logical_and( proba[l] > 0.5, seg == l )
                    # remove half of the correctly classified voxels
                    points = np.transpose(np.nonzero(correct))
                    if len(points) > 10:
                        np.random.shuffle(points)
                        points = points[:len(points)/2]
                        seg[points[:,0],
                            points[:,1],
                            points[:,2]] = 255

                knowledge = self.get_knowledge(img,proba,mask=mask)
                forest.add_image_autocontext(img,seg,knowledge)

                # irtk.imwrite( "debug/"+data['patient_id']+"_knowledge"+str(i)+".nii.gz",
                #               knowledge )                  

            print "starting to learn autocontext",i
            forest.grow( self.params['nb_samples'],
                         self.params['nb_background_samples'] )

            print "writing"
            forest.write(self.folder(i))
            print "done", i

            feature_importance = forest.get_feature_importance()
            mapping = self.feature_mapping()

            if len(feature_importance) != len(mapping):
                print "ERROR: forest.get_feature_importance() returns", len(feature_importance), "features"
                print "       feature_mapping() expects", len(mapping), "features"
            
            feature_importance = dict( zip(mapping,
                                           feature_importance) )
            self.info['feature_importance'].append( feature_importance )
            print feature_importance
            
            i += 1
            
            # release memory
            del forest
            gc.collect()

            previous_dice_score = dice_score
            print "scoring"
            dice_score = self.score(validation_patients,nb_autocontext=i)
            improvement = dice_score - previous_dice_score
            
            self.info['validation_scores'].append(dice_score)
            self.info['improvements'].append(improvement)
            
            if self.params['verbose']:
                print "Validation score:", dice_score
                print "improvement:", improvement

        self.save()
    
    def score( self,
               validation_patients,
               nb_autocontext=None ):
        gc.collect()
        filenames = []
        for patient_id in validation_patients:
            img_filename = self.params['img_folder'] + "/" + patient_id + self.params['file_extension']
            filenames.append(img_filename)
            
        # probas = Parallel(n_jobs=self.params['n_jobs'])(delayed(predict_level)( self,
        #                                                                         img_filename,
        #                                                                         all_ga[patient_id],
        #                                                                         level=level,
        #                                                                         nb_autocontext=nb_autocontext )
        #                                                 for patient_id,img_filename in zip(validation_patients,filenames) )

        probas = []
        for patient_id,img_filename in zip(validation_patients,filenames):
            print img_filename
            probas.append( predict( self,
                                    img_filename,
                                    nb_autocontext=nb_autocontext ) )

        print "will compute Dice scores"
        score = 0.0
        n = 0
        for patient_id,proba in zip(validation_patients,probas):
            header = proba.get_header()
            header['dim'][3] = 1
            if self.params['offline_preprocessing']:
                seg_filename =  "offline_preprocessing/"+patient_id+"_seg"+ self.params['file_extension']
            else:
                seg_filename =  self.params['seg_folder'] + "/" +patient_id+"_seg"+ self.params['file_extension']
            seg = irtk.imread( seg_filename, dtype="uint8" )
            #seg = seg.resample( self.params['resample'], interpolation="nearest").astype('uint8')

            # irtk.imwrite( "debug/"+patient_id+"_proba"+str(nb_autocontext)+".nii.gz",
            #                   proba ) 
                            
            # we skip mother/background as it depends of mask
            for i in [1]:#xrange(1,proba.shape[0]):
                dice,overlap = (seg==i).dice( self.hard_thresholding( proba[i] ),
                                              verbose=False)
                score += dice
                n += 1
        
        return score/n

    def email_log( self,
                   txt="",
                   me='kpk09@doc.ic.ac.uk', # the sender's email address
                   you='kevin.keraudren10@imperial.ac.uk', # the recipient's email address
                   smtp='smarthost.cc.ic.ac.uk' ):
        # Create a text/plain message
        txt = str(self)+"\n"+txt
        txt += "\n" + socket.gethostname()
        msg = MIMEText(txt)

        msg['Subject'] = "["+repr(self)+"] Testing: " + self.params['name']
        msg['From'] = me
        msg['To'] = you

        # Send the message via our own SMTP server, but don't include the
        # envelope header.
        s = smtplib.SMTP(smtp)
        s.sendmail(me, [you], msg.as_string())
        s.quit()

def _DetectorReduce(params):
    return OrganDetector(**params)

