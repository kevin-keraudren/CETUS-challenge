from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension( "simpleforest", ["simpleforest.pyx"],
                   language="c++",
                   include_dirs=get_numpy_include_dirs()+['src','.',"/vol/medic02/users/kpk09/LOCAL/include"],
                   library_dirs=["/vol/medic02/users/kpk09/LOCAL/lib"],
                   libraries=['opencv_core',
                              'opencv_imgproc',
                              'opencv_features2d',
                              'opencv_gpu',
                              'opencv_calib3d',
                              'opencv_objdetect',
                              'opencv_video',
                              'opencv_highgui',
                              'opencv_ml',
                              'opencv_legacy',
                              'opencv_contrib',
                              'opencv_flann',
                              'tbb',
                              'boost_filesystem-mt',
                              'boost_system-mt',
                              'boost_regex-mt'],
                   extra_link_args=["-w"],
                   extra_compile_args=["-w"]
                  ),
        Extension( "integralforest", ["integralforest.pyx"],
                   language="c++",
                   include_dirs=get_numpy_include_dirs()+['src','.',"/vol/medic02/users/kpk09/LOCAL/include"],
                   library_dirs=["/vol/medic02/users/kpk09/LOCAL/lib"],
                   libraries=['opencv_core',
                              'opencv_imgproc',
                              'opencv_features2d',
                              'opencv_gpu',
                              'opencv_calib3d',
                              'opencv_objdetect',
                              'opencv_video',
                              'opencv_highgui',
                              'opencv_ml',
                              'opencv_legacy',
                              'opencv_contrib',
                              'opencv_flann',
                              'tbb',
                              'boost_filesystem-mt',
                              'boost_system-mt',
                              'boost_regex-mt'],
                   extra_link_args=["-w"],
                   extra_compile_args=["-w"]
                  ),
        Extension("fastutils", ["fastutils.pyx"])
        ]
    )
