#!/usr/bin/python

# http://shambool.com/2012/05/01/classifying-points-by-random-trees-cv2rtrees-in-python/
# OpenCV-2.4.0/samples/python2/letter_recog.py
# http://fossies.org/dox/OpenCV-2.4.2/letter__recog_8py_source.html
# http://opencv.itseez.com/modules/ml/doc/random_trees.html#cvrtrees

import cv, cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.simpleforest import simpleForest
import math

test="linearND"

t = np.arange(0,10,0.1)

theta = [0,30,60]
colors = np.array( [[255,0,0],
                    [0,255,0],
                    [0,0,255]],
                   dtype='float' )


trainData = np.zeros((len(t)*len(theta),2),dtype='float32')
responses = np.zeros(len(t)*len(theta),dtype=int)

for c in range(len(theta)):
    trainData[c*len(t):(c+1)*len(t),0] = t**2*np.cos(t+theta[c]) # x
    trainData[c*len(t):(c+1)*len(t),1] = t**2*np.sin(t+theta[c]) # y
    responses[c*len(t):(c+1)*len(t)] = c

print len(responses),len(trainData)

forest = simpleForest(100,test=test)
forest.grow_classification( trainData, responses )

print "writing"
forest.write("test_forest")
print "done"
print "reading"
forest = simpleForest(folder="test_forest",test=test)
print "done"

img = np.zeros((512,512,3))
v_min = trainData.min()
v_max = trainData.max()
step = float(v_max - v_min)/img.shape[0]
grid = np.arange( v_min, v_max, step )

Y,X = np.mgrid[v_min:v_max:step,
               v_min:v_max:step]
Y = Y.ravel()
X = X.ravel()
XY = np.concatenate((np.reshape(X,(X.shape[0],1)),
                      np.reshape(Y,(Y.shape[0],1))),axis=1).astype('float64')

p = forest.predict_soft(XY)
img[((XY[:,1]-v_min)/step).astype('int'),
    ((XY[:,0]-v_min)/step).astype('int')] = np.dot( p, colors)

radius = 3
points = (trainData - v_min)/step
for p,r in zip(points,responses):
    cv2.circle(img, tuple(p), radius+1, (0,0,0), thickness=-1 )
    cv2.circle(img, tuple(p), radius, colors[r], thickness=-1 )

cv2.imwrite('simpleforest_classification.png',img)
exit(0)

# regression
nb_points = 50

# create data points
trainData = []
responses = []
for i in xrange(nb_points):
    x = 20*np.random.random() - 10
    y = 20*np.random.random() - 10
    trainData.append([y,x])
    responses.append([math.sqrt(x*x+y*y),x+y,x-y])

trainData = np.array(trainData,dtype="float")    
    
forest = simpleForest(100,test=test)
forest.grow_regression( trainData, responses )    

img = np.zeros((512,512,3), dtype="float")
v_min = -10 #trainData.min()
v_max = 10#trainData.max()
step = float(v_max - v_min)/img.shape[0]
grid = np.arange( v_min, v_max, step )

Y,X = np.mgrid[v_min:v_max:step,
               v_min:v_max:step]
Y = Y.ravel()
X = X.ravel()
XY = np.concatenate((np.reshape(X,(X.shape[0],1)),
                      np.reshape(Y,(Y.shape[0],1))),axis=1).astype('float64')

p = forest.predict_regression(XY)
p = np.array(p)
print p.shape, XY.shape
img[((XY[:,1]-v_min)/step).astype('int'),
    ((XY[:,0]-v_min)/step).astype('int')] = p

radius = 3
points = ((trainData - v_min)/step).astype('int32')
for p,r in zip(points,responses):
    cv2.circle(img, tuple(p), radius, r, thickness=-1 )

img -= img.min()
img /= img.max()
img *= 255

for p,r in zip(points,responses):
    cv2.circle(img, tuple(p), radius+1, (0,0,0), thickness=1 )
    
cv2.imwrite('simpleforest_regression.png',img)    
