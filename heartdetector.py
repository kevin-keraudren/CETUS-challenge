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

from skimage import morphology
from skimage.morphology import watershed
import scipy.ndimage as nd

from scipy.stats.mstats import mquantiles
import itertools

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
             'extra_layers' : np.array( [], dtype='float32' ),
             'metadata' : None,
             'mask' : mask }

def predict_autocontext( self,
                         img,
                         mask,
                         extra_layers,
                         metadata,
                         nb_labels,
                         ga,
                         nb_autocontext,
                         debug=False,
                         return_all=False ):
    proba = np.ones((nb_labels,img.shape[0],img.shape[1],img.shape[2]),dtype='float32')
    proba /= nb_labels

    header = img.get_header()
    header['dim'][3] = nb_labels
    proba = irtk.Image(proba,header)
    
    all_steps = []

    for k in xrange(nb_autocontext):
        metadata = self.get_center_axis(proba,k)
        knowledge = self.get_knowledge(img,proba,extra_layers,mask=mask)

        if debug:
            irtk.imwrite("knowledge_"+str(k)+".nii.gz", knowledge)

        forest = integralForest( folder=self.folder(k),
                                 test=self.params['test'],
                                 parallel=self.params['parallel'],
                                 nb_knowledge_layers=knowledge.shape[0] )
        proba = forest.predict_autocontext( img,
                                            knowledge,
                                            mask,
                                            self.params['ksampling'],
                                            metadata )
        proba = irtk.Image(proba,header)
        if return_all:
            all_steps.append( proba.copy() )
        if debug:
            irtk.imwrite("debug_"+str(k)+".nii.gz", proba)
        
        if k < 1:
            for i in xrange(proba.shape[0]):
                if i == 0:
                    proba[i] = 0
                else:
                    proba[i] = self.groups[i-1].get_center(proba[i])
                
        #     # volume constraint
        #     # set not ventricule to 0
        #     tmp_proba = proba[1]
        #     for i in xrange(proba.shape[0]):
        #         if i == 1:
        #             continue
        #         proba[i] = 0
                
        #     # rescale ventricule
        #     target_volume = 182950.0*0.001**3
        #     #target_volume = 151807.0*0.001**3
            
        #     if k == 0:
        #         target_volume *= 0.5
        #     # elif k == 1:
        #     #     target_volume *= 0.25
        #     # elif k == 2:
        #     #     target_volume *= 0.5
                
        #     box_volume = float(proba.shape[1])*proba.header['pixelSize'][2]*float(proba.shape[2])*proba.header['pixelSize'][1]*float(proba.shape[3])*proba.header['pixelSize'][0]

        #     ratio = float(target_volume) / float(box_volume)

        #     #print "ratio", ratio
        #     q0 = mquantiles( tmp_proba.flatten(), prob=[1.0-ratio] )
        #     tmp_proba[proba[1]<q0] = q0
        #     tmp_proba -= tmp_proba.min()
        #     tmp_proba /= tmp_proba.max()

        #     lcc = irtk.largest_connected_component(tmp_proba,fill_holes=False)
        #     tmp_proba[lcc==0] = 0

        #     proba[1] = tmp_proba
            
        if debug:
            print "done autocontext", k
        #     irtk.imwrite("debug_rescaled_"+str(k)+".nii.gz", proba)

    if not return_all:
        return proba
    else:
        return all_steps

def predict( self,
             filename,
             ga,
             nb_autocontext=None,
             mask=None,
             debug=False,
             return_all=False ):
    nb_labels = len(self.labels)+1
    
    if nb_autocontext is None:
        nb_autocontext = len(glob(self.params['name'] + "_*"))

    img = irtk.imread( filename, dtype="float32" )
    img = img.resample( pixelSize=self.params['resample'], interpolation='linear' ).rescale(0,1000)

    extra_layers = []

    if mask is None:
        mask = irtk.ones( img.get_header(), dtype="uint8" )
        mask[img==0] = 0
    else:
        mask = mask.resample( pixelSize=self.params['resample'], interpolation='nearest' ).astype('uint8')

    metadata = None
    
    probas = predict_autocontext( self,
                                  img,
                                  mask,
                                  np.array( extra_layers, dtype="float32" ),
                                  metadata,
                                  nb_labels,
                                  ga,
                                  nb_autocontext,
                                  debug=debug,
                                  return_all=return_all )
    
    return probas

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
        # if os.path.basename(f).split('_')[0] in training_patients:
        #     training_files.append(f)
        # else:
        #     testing_files.append(f)
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

def feature_mapping( test,
                     groups,
                     use_extra_layers,
                     use_background_distance=False,
                     metadata_mapping=["ga"] ):
    if use_background_distance:
        use_extra_layers = ["Background"] + use_extra_layers
    if test == "autocontext":
        return ["Image"] + ["Proba "+g.name for g in groups] + ["Distance "+g.name for g in groups] + use_extra_layers
    if test == "autocontext2" or test == "autocontextN":
        l = ["Proba "+g.name for g in groups] + ["Distance "+g.name for g in groups]
        l = [ x+" / "+y for x,y in itertools.product(l, repeat=2) ]
        return ["Image"] + l
    elif test == "autocontextDistancePrior":
        return ["Distance"]+ feature_mapping( "autocontext",
                                              groups,
                                              use_extra_layers,
                                              use_background_distance,
                                              metadata_mapping )
    elif test == "autocontextMetadata":
        return feature_mapping( "autocontext",
                                groups,
                                use_extra_layers,
                                use_background_distance,
                                metadata_mapping ) + metadata_mapping
    elif test == "autocontextGradient":
        return ["dxdydz"]+ feature_mapping( "autocontext",
                                            groups,
                                            [],
                                            use_background_distance,
                                            metadata_mapping )
    elif test == "heartautocontext":
        return ["r","z"]+ feature_mapping( "autocontext",
                                            groups,
                                            [],
                                            use_background_distance,
                                            metadata_mapping )
    elif test == "autocontextGradientDistancePrior":
        return ["Distance"]+["dxdydz"]+ feature_mapping( "autocontext",
                                            groups,
                                            [],
                                            use_background_distance,
                                            metadata_mapping )
    else:
        raise ValueError("Unknown test")


class Group(object):
    def __init__( self, labels, name="X" ):
        self.labels = labels
        self.name = name

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
        res = irtk.zeros( proba.get_header() )
        tmp = proba.view(np.ndarray).copy()
        tmp[self.hard_thresholding(proba)==0] = 0
        if tmp.sum() == 0:
            return res
        center = np.array( nd.center_of_mass( tmp ), dtype='int' )
        res[center[0],center[1],center[2]] = 1
        return res

class HeartDetector(object):
    labels = [ "left_ventricule",
               "wall",
               "valves" ]
    groups = [ Group( "left_ventricule", name="left_ventricule" ),
               Group( "wall", name="wall" ),
               Group( "valves", name="valves" )]

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
                  use_extra_layers=[],
                  use_background_distance=True,
                  use_world_align=True,
                  resample=np.array([2,2,2,1],dtype='float'),
                  lambda_gdt=100 ):
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
                        "hull_folder" : hull_folder,
                        "name" : name,
                        "nb_samples" : nb_samples,
                        "nb_background_samples" : nb_background_samples,
                        "use_extra_layers" : use_extra_layers,
                        "use_background_distance" : use_background_distance,
                        "use_world_align" : use_world_align,
                        "resample" : resample,
                        "lambda_gdt" : lambda_gdt }

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
        return "HeartDetector"       

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
    
    def read_ga(self):
        all_ga = {}
        all_patients = get_patients(self.params['seg_folder'])
        for p in  all_patients:
            all_ga[p] = 0.0
        return all_ga

    def get_knowledge( self, img, proba, extra_layers, mask=None ):
        knowledge = proba[1:].rescale(0,1000).view(np.ndarray)
        if len(knowledge.shape) == 3:
            knowledge = knowledge[np.newaxis,...]

        group_distances = Parallel(n_jobs=self.params['n_jobs'])(delayed(gdt)( img,
                                                                               self.groups[group_id-1].get_center(proba[group_id]),
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

        if len(extra_layers) > 0:
            knowledge = np.concatenate((knowledge,extra_layers), axis=0)
        
        header = img.get_header()
        header['dim'][3] = knowledge.shape[0]
        knowledge = irtk.Image(knowledge.copy(order='C').astype('float32'),header)
        knowledge = knowledge.resample(img.header['pixelSize'][0]*self.params['ksampling'])

        # irtk.imwrite("debug.nii.gz",knowledge)
        # exit(0)

        return knowledge    

    def get_center_axis(self,proba,nb_autocontext):
        # FIXME
        return np.zeros(6,dtype='float64')
    
        if nb_autocontext==0:
            return np.array( [float(proba.shape[1])/2,
                              float(proba.shape[2])/2,
                              float(proba.shape[3])/2,
                              1,0,0],
                             dtype="float64" )
        else:
            heart_seg = self.groups[0].hard_thresholding( proba[1] )
            points = np.transpose(np.nonzero(heart_seg))[::4]
            center,S,V = fit_ellipsoidPCA( points )
            if V[0,0] < 0:
                V *= -1
            return np.array( [center[0],center[1],center[2],
                              V[0,0],V[0,1],V[0,2]],
                             dtype="float64" )

    def do_offline_preprocessing( self, patient_ids, n_jobs ):
        Parallel(n_jobs=n_jobs)(delayed(preprocess_training_data)( patient_id,
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
        nb_labels = len(self.labels)+1
        ## Preprocess data only once to speed up training
        ## (requires more memory)

        # split patients to get validation set
        np.random.shuffle(patient_ids)

        n_validation = min(len(patient_ids)/2,n_validation)
        
        training_patients = patient_ids[:-n_validation]
        validation_patients = patient_ids[-n_validation:]

        self.info['training_patients'] = training_patients
        self.info['validation_patients'] = validation_patients

        all_ga = self.read_ga()

        if self.params['verbose']:
            print "fitting with", len(training_patients), "training patients and", \
                len(validation_patients), "validation patients"

            print "doing preprocessing..."
        gc.collect()
        training_data = Parallel(n_jobs=self.params['n_jobs'])(delayed(preprocess_training_data)( patient_id,
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
                                     nb_knowledge_layers=2*(nb_labels-1)+len(training_data[0]['extra_layers']),
                                     ksampling=self.params['ksampling'],
                                     verbose=False,
                                     nfeatures=len(feature_mapping( self.params['test'],
                                                                    self.groups,
                                                                    self.params['use_extra_layers'],
                                                                    self.params['use_background_distance']))
                                     )

            print "predicting training data"
            tmp_probas = Parallel(n_jobs=self.params['n_jobs'])(delayed(predict_autocontext)( self,
                                                                                              data['img'],
                                                                                              data['mask'],
                                                                                              data['extra_layers'],
                                                                                              data['metadata'],
                                                                                              nb_labels,
                                                                                              all_ga[data['patient_id']],
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
                extra_layers = data['extra_layers']
                metadata = data['metadata']
                ga = all_ga[data['patient_id']]
                #print data['patient_id'], metadata,img.shape

                metadata = self.get_center_axis(proba,i)

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

                knowledge = self.get_knowledge(img,proba,extra_layers,mask=mask)
                forest.add_image_autocontext(img,seg,knowledge,metadata)

                # irtk.imwrite( "debug/"+data['patient_id']+"_knowledge"+str(i)+".nii.gz",
                #               knowledge )                  

            print "starting to learn autocontext",i
            forest.grow( self.params['nb_samples'],
                         self.params['nb_background_samples'] )

            print "writing"
            forest.write(self.folder(i))
            print "done", i

            feature_importance = forest.get_feature_importance()
            mapping = feature_mapping(self.params['test'],
                                      self.groups,
                                      self.params['use_extra_layers'],
                                      self.params['use_background_distance'])

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
            img_filename = self.params['img_folder'] + "/" + patient_id + ".nii.gz"
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
                                          0.0,
                                          nb_autocontext=nb_autocontext ) )

        print "will compute Dice scores"
        score = 0.0
        n = 0
        for patient_id,proba in zip(validation_patients,probas):
            header = proba.get_header()
            header['dim'][3] = 1
            seg_filename =  self.params['seg_folder'] +"/"+patient_id+"_seg.nii.gz"
            seg = irtk.imread( seg_filename, dtype="uint8" )
            seg = seg.resample( self.params['resample'], interpolation="nearest").astype('uint8')

            # irtk.imwrite( "debug/"+patient_id+"_proba"+str(nb_autocontext)+".nii.gz",
            #                   proba ) 
                            
            # we skip mother/background as it depends of mask
            for i in [1]:#xrange(1,proba.shape[0]):
                # dice,overlap = (seg==i).dice( proba[i] > 0.5,
                #                               verbose=False)
                dice,overlap = (seg==i).dice( self.groups[0].hard_thresholding( proba[i] ),
                                              verbose=False)
                score += dice
                n += 1
        
        return score/n

    def email_log(self, txt=""):
        # Create a text/plain message
        txt = str(self)+"\n"+txt
        txt += "\n" + socket.gethostname()
        msg = MIMEText(txt)

        me = 'kpk09@doc.ic.ac.uk' # the sender's email address
        you = 'kevin.keraudren10@imperial.ac.uk' # the recipient's email address
        msg['Subject'] = "[heartdetector] Testing: " + self.params['name']
        msg['From'] = me
        msg['To'] = you

        # Send the message via our own SMTP server, but don't include the
        # envelope header.
        s = smtplib.SMTP('smarthost.cc.ic.ac.uk')
        s.sendmail(me, [you], msg.as_string())
        s.quit()

def _DetectorReduce(params):
    return HeartDetector(**params)

if __name__ == "__main__":
    params = { 'img_folder':"/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Training/input_data",
               'seg_folder':"/vol/biomedic/users/kpk09/DATASETS/CETUS_data/Training/input_data",
               'ga_files':[],
               'name':"debug_forest_border",
               'nb_samples':200,
               'nb_background_samples':400,
               'nb_tests':100,
               'verbose':True,
               'n_jobs':5,
               'parallel':-1,
               'test':"autocontext",
               #'test':"autocontextGradient",
               #'test':"autocontextDistancePrior",
               #'use_extra_layers':["slic_entropy","gradient","maximum","minimum"],
               #'use_extra_layers':["dx","dy","dz"],
               'use_extra_layers':[],
               'use_background_distance':False,
               'use_world_align':True,
               'resample':np.array([0.001,0.001,0.001,1],dtype='float32'),
               'ksampling':1.0,
               'dx':50,
               'dy':50,
               'dz':50,
               'cx':50,
               'cy':50,
               'cz':50
               }
    detector = HeartDetector( **params )
    print detector
    
    all_patients = sorted(get_patients(detector.params['seg_folder']))
    print all_patients,len(all_patients)

    n_testing = 3
    training_patients = all_patients[:-n_testing]
    testing_patients = all_patients[-n_testing:]

    #training_patients, testing_patients = split_patients(all_patients,3)
    
    print training_patients
    detector.fit( training_patients[:5],
                  max_autocontext=10,
                  min_dice_score=0.99,
                  n_validation=10 )

    #print detector.info

    print "will predict testing data..."
    testing_score = detector.score(testing_patients,nb_autocontext=len(detector.info['improvements'])-1)
    print testing_score
    detector.email_log("Testing score: " + str(testing_score))
