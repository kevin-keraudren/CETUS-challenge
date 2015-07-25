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
from irtk.ext.slic import slic
from irtk.ext.FitEllipsoid import get_voxels

__all__ = [ "world_align",
            "edt",
            "gdt",
            "background_distance",
            "get_background",
            "slic_feature" ]

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

def get_entropy(img):
    """
    http://stackoverflow.com/questions/16647116/faster-way-to-analyze-each-sub-window-in-an-image
    """
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = hist.ravel()/hist.sum()
    logs = np.log2(hist+0.00001)
    entropy = -1 * (hist*logs).sum()
    return entropy

def slic_feature(img, size=300, compactness=10, sigma=2.0, feature="entropy"):
    #img = img.saturate(0.01,0.9).rescale().astype("float32")
    labels = slic( img.astype("float32").gaussianBlurring(sigma),
                   size=size,
                   compactness=compactness )
    max_label = labels.max()
    voxels = get_voxels(labels,max_label)
    res = irtk.zeros(img.get_header(),dtype='float32')
    if feature == "entropy":
        img = img.rescale().astype('uint8')
        for l in xrange(max_label+1):
            if voxels[l].shape[0] == 0:
                continue
            values = img[voxels[l][:,0],
                         voxels[l][:,1],
                         voxels[l][:,2]]
            vmax = values.max()
            if vmax == 0:
                continue
            vmin = values.min()
            if vmax == vmin:
                continue
            res[voxels[l][:,0],
                voxels[l][:,1],
                voxels[l][:,2]] = get_entropy(values)
    elif feature == "std":
        for l in xrange(max_label+1):
            if voxels[l].shape[0] == 0:
                continue
            res[voxels[l][:,0],
                voxels[l][:,1],
                voxels[l][:,2]] = img[voxels[l][:,0],
                                      voxels[l][:,1],
                                      voxels[l][:,2]].std()            
    else:
        raise ValueError("Unknown feature:"+feature)

    return irtk.Image(res,img.get_header())

def slic_std(img, size=300, compactness=10, sigma=2.0):
    #img = img.saturate(0.01,0.9).rescale().astype("float32")
    labels = slic( img.astype("float32").gaussianBlurring(sigma),
                   size=size,
                   compactness=compactness )
    max_label = labels.max()
    voxels = get_voxels(labels,max_label)
    res = irtk.zeros(img.get_header(),dtype='float32')
    img = img.rescale().astype('uint8')
    for l in xrange(max_label+1):
        if voxels[l].shape[0] == 0:
            continue
        values = img[voxels[l][:,0],
                     voxels[l][:,1],
                     voxels[l][:,2]]
        vmax = values.max()
        if vmax == 0:
            continue
        vmin = values.min()
        if vmax == vmin:
            continue
        res[voxels[l][:,0],
            voxels[l][:,1],
            voxels[l][:,2]] = get_entropy(values)

    return irtk.Image(res,img.get_header())

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

    # irtk.imwrite( "slicEntropy.nii.gz", slic_feature( img,
    #                                                   feature="entropy",
    #                                                   size=float(img.shape[0]*img.shape[1]*img.shape[2])/500,
    #                                                   compactness=10))
                  
    # irtk.imwrite( "slicSTD.nii.gz", slic_feature( img,
    #                                               feature="std",
    #                                               size=float(img.shape[0]*img.shape[1]*img.shape[2])/500,
    #                                               compactness=10))
