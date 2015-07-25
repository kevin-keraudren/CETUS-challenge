import numpy as np
cimport numpy as np
import six
import copy

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string

np.import_array()

cdef extern from "SimpleForest.h":
    cdef cppclass SimpleForest:
        SimpleForest( int,
                      double,
                      int,
                      int,
                      int,
                      int,
                      string,
                      bool )
        SimpleForest( string, string )
        void grow_classification( vector[vector[float]]&, vector[int]& )
        void grow_regression( vector[vector[float]]&, vector[vector[double]]& )
        vector[int] predict_hard( vector[vector[float]]& )
        vector[vector[double]] predict_soft( vector[vector[float]]& )
        vector[vector[double]] predict_regression( vector[vector[float]]& )
        void write( string folder )
        void set_bagging( double b )
        void set_verbose( bool verbose )
        void set_ntrees( int n )
        void set_test( string test )
        void set_nb_tests( int n )
        void set_min_items( int n)
        void set_max_depth( int n)
        
cdef class simpleForest:
    cdef SimpleForest *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self,
                  int ntrees=10,
                  string folder="",
                  double bagging=0.7,
                  int max_depth=10,
                  int min_items=20,
                  int nb_tests=1000,
                  int parallel=-1,
                  string test="axis",
                  bool verbose=True ):
        if len(folder) > 0:
            self.thisptr = new SimpleForest( folder, test )
        else:
            self.thisptr = new SimpleForest( ntrees,
                                             bagging,
                                             max_depth,
                                             min_items,
                                             nb_tests,
                                             parallel,
                                             test,
                                             verbose )
    def __dealloc__(self):
        del self.thisptr
    def grow_classification(self, vector[vector[float]] points, vector[int] responses):
        self.thisptr.grow_classification(points,responses)
    def grow_regression(self, vector[vector[float]] points, vector[vector[double]] responses):
        self.thisptr.grow_regression(points,responses)        
    def predict_hard(self, vector[vector[float]] points):
        return self.thisptr.predict_hard(points)
    def predict_soft(self, vector[vector[float]] points):
        return self.thisptr.predict_soft(points)
    def predict_regression(self, vector[vector[float]] points):
        return self.thisptr.predict_regression(points)        
    def write(self, string folder):
        self.thisptr.write(folder)

cdef class RegressionForest:
    cdef SimpleForest *thisptr      # hold a C++ instance which we're wrapping
    cdef public object params
    def __cinit__(self,
                  int n_estimators=10,
                  string folder="",
                  double bootstrap=0.7,
                  int max_depth=10,
                  int min_items=20,
                  int nb_tests=1000,
                  int parallel=-1,
                  string test="axis",
                  bool verbose=True ):
        self.params = { "n_estimators" : n_estimators,
                        "bootstrap" : bootstrap,
                        "verbose" : verbose,
                        "max_depth" : max_depth,
                        "min_items" : min_items,
                        "nb_tests" : nb_tests,
                        "parallel" : -1,
                        "test" : test
                        }
        if len(folder) > 0:
            self.thisptr = new SimpleForest( folder, test )
        else:
            self.thisptr = new SimpleForest( n_estimators,
                                             bootstrap,
                                             max_depth,
                                             min_items,
                                             nb_tests,
                                             parallel,
                                             test,
                                             verbose )

    def __dealloc__(self):
        del self.thisptr
    def __reduce__(self):
        """
        Required for pickling/unpickling, which is used for instance
        in joblib Parallel.
        An example implementation can be found in numpy/ma/core.py .
        """        
        return ( _ForestReduce, (self.params,) )        
    def fit(self, vector[vector[float]] points, vector[vector[double]] responses):
        self.thisptr.grow_regression(points,responses)        
    def predict(self, vector[vector[float]] points):
        return self.thisptr.predict_regression(points)        
    def write(self, string folder):
        self.thisptr.write(folder)
    def set_params(self, **params):
        for key, value in six.iteritems(params):
            self.params[key] = value
            if key == "bootsrap":
                self.thisptr.set_bagging(value)
            elif key == "verbose":
                self.thisptr.set_verbose(value)
            elif key == "n_estimators":
                self.thisptr.set_ntrees(value)
            elif key == "test":
                self.thisptr.set_test(value)
            elif key == "nb_tests":
                self.thisptr.set_nb_tests(value)
            elif key == "min_items":
                self.thisptr.set_min_items(value)
            elif key == "max_depth":
                self.thisptr.set_max_depth(value)
                
    def get_params(self,deep=False):
        if not deep:
            return self.params
        else:
            return copy.deepcopy(self.params)

def _ForestReduce(params):
    return RegressionForest(**params)

