#!/usr/bin/python

import cv, cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.integralforest import integralForest
from glob import glob
import re

import irtk

forest = integralForest( ntrees=5,
                         bagging=0.5,
                         max_depth=7,
                         min_items=10,
                         nb_tests=1000,
                         parallel=-1,
                         test="patch",
                         cx=50,  cy=50,  cz=0,
                         dx=50,  dy=50,  dz=0 )

images = []
segmentations = []

raw_data_folder = "/home/kevin/Imperial/PhD/gitlab/ess/brain_test/raw_data/"
ground_truth = glob("/home/kevin/Imperial/PhD/gitlab/ess/brain_test/ground_truth/*")#[:50]

segmentations = []
for f in ground_truth:
    pattern =  r'/(?P<raw_file>[^/]+)_(?P<x>\d+)_(?P<y>\d+)_(?P<w>\d+)_(?P<h>\d+)\.png$'
    match = re.search( pattern, f )
    if match is None:
        print "Cannot parse: " + f
        sys.exit(1)
    m = match.groupdict()

    x = int(m['x'])
    y = int(m['y'])
    w = int(m['w'])
    h = int(m['h'])

    f = raw_data_folder + m['raw_file']
    print f
    img = cv2.imread( f, 0 ).astype('float32')
    #img[img<20] = 0
    seg = np.zeros(img.shape,dtype='int32')
    seg[y+h/2-20:y+h/2+20,x+w/2-20:x+w/2+20] = 1
    #seg[y+h/2+100:y+h/2+140,x+w/2-60:x+w/2+60] = 0
    
    # cv2.imwrite("img.png",img.astype('uint8'))
    # cv2.imwrite("seg.png",seg*255)
    # exit(0)
    
    img = img[np.newaxis,...].copy()
    seg = seg[np.newaxis,...].copy()
    
    segmentations.append(seg)
    forest.add_image(img,seg)#segmentations[-1])
    # print img.max(),img.shape
    
    # res = forest.debug_mean(img.copy())
    # print res.max()
    # res = np.reshape(res,(seg.shape[1],seg.shape[2]))
    # res /= res.max()
    # res *= 255
    # cv2.imwrite("mean.png",res.astype('uint8'))
    # exit(0)

forest.grow( 100 )

# print "writing"
# forest.write("test_forest")
# print "done"
# print "reading"
# forest = integralForest(folder="test_forest")
# print "done"

img = cv2.imread( "378.png",0).astype("float32")
img = img[np.newaxis,...].copy()
res = forest.predict_hard(img)
res = np.squeeze(res)#.copy()
irtk.imshow(irtk.Image(img),seg=res,filename="overlay_seg.png")

res /= res.max()
res *= 255
cv2.imwrite("res_hard.png",res.astype('uint8'))

res = forest.predict_soft(img)
res = np.squeeze(res)
#res /= res.max()

res *= 255
irtk.imshow(irtk.Image(img),overlay=res.astype('uint8'),colors='jet',filename="overlay.png")


cv2.imwrite("res_proba.png",res.astype('uint8'))
