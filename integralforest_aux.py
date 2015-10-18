#!/usr/bin/python

import sys
import numpy as np
import scipy.ndimage as nd
import cv2

from  scipy.ndimage.measurements import label

from lib.fastutils import geodesic_distance_transform
from numpy.core.umath import logical_not
import joblib
from joblib import Parallel, delayed

import irtk

__all__ = [ "world_align",
            "edt",
            "gdt",
            "background_distance",
            "get_background" ]

def world_align(img,pixelSize=[3,3,3,1],interpolation='linear'):
    x_min, y_min, z_min, x_max, y_max, z_max = img.wbox()
    
    x0 = ( x_min + x_max ) / 2
    y0 = ( y_min + y_max ) / 2
    z0 = ( z_min + z_max ) / 2
    dx = int((x_max-x_min)/pixelSize[0])
    dy = int((y_max-y_min)/pixelSize[1])
    dz = int((z_max-z_min)/pixelSize[2])
    header = irtk.new_header( pixelSize=pixelSize,
                              orientation=np.eye( 3, dtype='float64' ),
                              origin=[x0,y0,z0,0],
                              dim=[dx,dy,dz,1])

    res = img.transform(target=header,interpolation=interpolation)
    return res.bbox(crop=True)

def edt( img, mask ):
    if mask.sum() == 0:
        return irtk.zeros(img.get_header())
    voxelSpacing = img.header['pixelSize'][:3][::-1]
    distanceMap = nd.distance_transform_edt( logical_not(mask),
                                             sampling=voxelSpacing)
    distanceMap -= nd.distance_transform_edt( mask,
                                              sampling=voxelSpacing)
    return irtk.Image(distanceMap,img.get_header())

def _geodesic_distance_transform( binaryImage,
                                  imageMagnitudes,
                                  numIterations=3,
                                  spacing=(1.0, 1.0, 1.0),
                                  includeEDT=True ):
    return geodesic_distance_transform( binaryImage,
                                        imageMagnitudes,
                                        numIterations,
                                        spacing,
                                        includeEDT )
    

def gdt( img, mask, includeEDT=True, l=1.0 ):
    if mask.sum() == 0:
        return irtk.zeros(img.get_header())
    voxelSpacing = img.header['pixelSize'][:3][::-1]
    grad =  irtk.Image( nd.gaussian_gradient_magnitude(img, 0.5),
                        img.get_header() )
    #irtk.imwrite("gradBefore.nii.gz",grad)
    grad = l*grad.saturate().rescale(0.0,1.0).as3D()
    #irtk.imwrite("gradAfter.nii.gz",grad)
    # distanceMap = geodesic_distance_transform( mask,
    #                                            grad,
    #                                            numIterations=3,
    #                                            spacing=voxelSpacing,
    #                                            includeEDT=includeEDT )
    # distanceMap -= geodesic_distance_transform( logical_not(mask),
    #                                             grad,
    #                                             numIterations=3,
    #                                             spacing=voxelSpacing,
    #                                             includeEDT=includeEDT )
    # return irtk.Image(distanceMap,img.get_header())

    # distanceMaps = Parallel(n_jobs=-1)(delayed(_geodesic_distance_transform)( m,
    #                                                                          grad,
    #                                                                          numIterations=3,
    #                                                                          spacing=voxelSpacing,
    #                                                                          includeEDT=includeEDT )
    #                                                              for m in [mask,
    #                                                                        logical_not(mask)]
    #                                                                          )
    # res = irtk.Image(distanceMaps[0]-distanceMaps[1],img.get_header())

    res = irtk.Image( _geodesic_distance_transform( mask,
                                                    grad,
                                                    numIterations=3,
                                                    spacing=voxelSpacing,
                                                    includeEDT=includeEDT ),
                      img.get_header() ).as3D()

    return res



def get_background(img):
    tmp_img = img.saturate(0.1,nonzero=True).rescale()
    labels,nb_labels = nd.label(tmp_img==0)

    sizes = np.bincount( labels.flatten(),
                         minlength=nb_labels+1 )
    sizes[0] = 0
    best_label = np.argmax( sizes )
    
    return labels == best_label
    
def background_distance(img,metric='geodesic',includeEDT=True):
    background = get_background(img)

    if metric == "euclidean":
        distanceMap = edt( img, background )
    elif metric == "geodesic":
        distanceMap = gdt( img, background, includeEDT )
    else:
        raise ValueError("Unknown metric: "+ metric)
    
    return irtk.Image(distanceMap,img.get_header())


if __name__ == "__main__":

    img = irtk.imread( sys.argv[1], dtype="float64" )
    #filtered = nd.minimum_filter(img,5)
    filtered = nd.gaussian_gradient_magnitude(img,0.5)
    img = irtk.Image(filtered,img.get_header())
    irtk.imwrite("test2.nii.gz",img)
    exit(0)
    
    img = world_align(img,pixelSize=[2,2,2,1])

    irtk.imwrite("distanceEDT.nii.gz",background_distance(img,metric="euclidean"))

    irtk.imwrite( "distanceGDT.nii.gz", background_distance(img,metric="geodesic"))
