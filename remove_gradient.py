#!/usr/bin/python

import irtk
import sys
import numpy as np

#from glob import glob

input_file = sys.argv[1]
output_file = sys.argv[2]

img = irtk.imread( input_file, dtype='float32' )
tmp = img.copy()

for z in range(img.shape[0]):
    s = img[z].copy()
    m = s[np.nonzero(s)].mean()
    if m == 0:
        continue
    #print s[np.nonzero(s)].mean()
    img[z][np.nonzero(s)] -=  m

img[tmp==0] = img.min()
img = img.rescale(0,1000)

irtk.imwrite(output_file, img )
