#!/usr/bin/python

import sys,os
import irtk
import numpy as np

def rand(scale):
    return float(scale)*(np.random.random(1)-0.5)*2

img = irtk.imread(sys.argv[1])
seg = irtk.imread(sys.argv[2])
prefix = sys.argv[3]

tx,ty,tz = img.ImageToWorld( [(img.shape[2]-1)/2,
                              (img.shape[1]-1)/2,
                              (img.shape[0]-1)/2] )
centering = irtk.RigidTransformation( tx=-tx, ty=-ty, tz=-tz )

t = irtk.RigidTransformation( tx=rand(img.header['pixelSize'][0]*10),
                              ty=rand(img.header['pixelSize'][1]*10),
                              tz=rand(img.header['pixelSize'][2]*10),
                              rx=rand(30),
                              ry=rand(30),
                              rz=rand(30) )
print t
t = centering.invert()*t*centering

new_img = img.transform(t,target=img.get_header(),interpolation='linear')
new_seg = seg.transform(t,target=img.get_header(),interpolation='nearest')

irtk.imwrite( "data_augmentation_"+prefix+"_"+os.path.basename(sys.argv[1]), new_img )
irtk.imwrite( "data_augmentation_"+prefix+"_"+os.path.basename(sys.argv[2]), new_seg )
