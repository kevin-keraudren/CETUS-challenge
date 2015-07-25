import numpy as np
cimport numpy as np

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string

np.import_array()

ctypedef float pixeltype
ctypedef float probatype
ctypedef float knowledgetype

cdef extern from "IntegralForest.h":
    cdef cppclass IntegralForest:
        IntegralForest( int ntrees,
                        double bagging,
                        int max_depth,
                        int min_items,
                        int nb_tests,
                        int parallel,
                        string test,
                        bool verbose,
                        int cx, int cy, int cz,
                        int dx, int dy, int dz,
                        int nb_labels,
                        int nb_knowledge_layers,
                        double ksampling,
                        int nfeatures )
        IntegralForest( string folder,
                        string test,
                        int parallel )
        void add_image( pixeltype* img,
                        int shape0,
                        int shape1,
                        int shape2,
                        unsigned char* seg )
        void add_image_autocontext( pixeltype* img,
                                    int shape0,
                                    int shape1,
                                    int shape2,
                                    knowledgetype* knowledge,
                                    int _nb_knowledge_layers,
                                    int kshape0,
                                    int kshape1,
                                    int kshape2,
                                    unsigned char* _seg,
                                    vector[double] metadata )
        void grow( int nb_samples,
                   int nb_background_samples )
        void predict_hard( pixeltype* img,
                           unsigned char* mask,
                           int shape0,
                           int shape1,
                           int shape2,
                           unsigned char* res )
        void predict_soft( pixeltype* img,
                           unsigned char* mask,
                           int shape0,
                           int shape1,
                           int shape2,
                           int _nb_labels,
                           probatype* res )
        void predict_autocontext( pixeltype* img,
                                  unsigned char* mask,
                                  int shape0,
                                  int shape1,
                                  int shape2,
                                  knowledgetype* knowledge,
                                  int _nb_knowledge_layers,
                                  int kshape0,
                                  int kshape1,
                                  int kshape2,
                                  double ksampling,
                                  vector[double] metadata,
                                  knowledgetype* res )
        void write( string folder )
        int get_nb_labels()
        void debug_mean( int n,
                         pixeltype* res )
        void debug_mean_knowledge( int n,
                                   knowledgetype* res )
        vector[double] get_feature_importance()
        
cdef class integralForest:
    cdef IntegralForest *thisptr      # hold a C++ instance which we're wrapping
    cdef int nb_labels
    cdef int nb_knowledge_layers
    
    def __cinit__( self,
                   int ntrees=10,
                   string folder="",
                   double bagging=0.7,
                   int max_depth=10,
                   int min_items=20,
                   int nb_tests=1000,
                   int parallel=-1,
                   string test="block",
                   bool verbose=True,
                   int cx=50, int cy=50, int cz=50,
                   int dx=50, int dy=50, int dz=50,
                   int nb_labels=2,
                   int nb_knowledge_layers=2,
                   double ksampling=1.0,
                   int nfeatures=1 ):
        if len(folder) > 0:
            self.thisptr = new IntegralForest( folder,
                                               test,
                                               parallel )
            self.nb_labels = self.thisptr.get_nb_labels()
            self.nb_knowledge_layers = nb_knowledge_layers
        else:
            self.thisptr = new IntegralForest( ntrees,
                                               bagging,
                                               max_depth,
                                               min_items,
                                               nb_tests,
                                               parallel,
                                               test,
                                               verbose,
                                               cx, cy, cz,
                                               dx, dy, dz,
                                               nb_labels,
                                               nb_knowledge_layers,
                                               ksampling,
                                               nfeatures )

    def __dealloc__(self):
        del self.thisptr

    def get_nb_labels(self):
        return self.thisptr.get_nb_labels()
        
    def add_image(self, np.ndarray[pixeltype, ndim=3,  mode="c"] img,
                  np.ndarray[unsigned char, ndim=3,  mode="c"] seg):
        cdef int shape0 = img.shape[0]
        cdef int shape1 = img.shape[1]
        cdef int shape2 = img.shape[2]
        self.thisptr.add_image( <pixeltype*> img.data,
                                 shape0,
                                 shape1,
                                 shape2,
                                 <unsigned char*> seg.data )
    def add_image_autocontext(self, np.ndarray[pixeltype, ndim=3,  mode="c"] img,
                              np.ndarray[unsigned char, ndim=3,  mode="c"] seg,
                              np.ndarray[knowledgetype, ndim=4,  mode="c"] knowledge,
                              np.ndarray[double, ndim=1,  mode="c"] metadata=np.zeros(0,dtype='float64') ):
        cdef int shape0 = img.shape[0]
        cdef int shape1 = img.shape[1]
        cdef int shape2 = img.shape[2]
        cdef int _nb_knowledge_layers = knowledge.shape[0]
        cdef int kshape0 = knowledge.shape[1]
        cdef int kshape1 = knowledge.shape[2]
        cdef int kshape2 = knowledge.shape[3]
        cdef vector[double] vect_metadata = metadata
        self.thisptr.add_image_autocontext( <pixeltype*> img.data,
                                             shape0,
                                             shape1,
                                             shape2,
                                             <knowledgetype*> knowledge.data,
                                             _nb_knowledge_layers,
                                             kshape0,
                                             kshape1,
                                             kshape2,
                                             <unsigned char*> seg.data,
                                             vect_metadata )        

    def debug_mean( self, np.ndarray[pixeltype, ndim=3,  mode="c"] img ):
        cdef int shape0 = img.shape[0]
        cdef int shape1 = img.shape[1]
        cdef int shape2 = img.shape[2]
        cdef np.ndarray[unsigned char, ndim=3,  mode="c"] seg = np.zeros( (shape0,
                                                                 shape1,
                                                                 shape2),
                                                                dtype='uint8' )
        self.add_image(img,seg)

        cdef np.ndarray[pixeltype, ndim=3,  mode="c"] res = np.zeros( (shape0,
                                                                 shape1,
                                                                 shape2),
                                                                dtype='float32' )
        self.thisptr.debug_mean( 0, <pixeltype*> res.data )
        return res

    def debug_mean_knowledge( self,
                              np.ndarray[pixeltype, ndim=3,  mode="c"] img,
                              np.ndarray[knowledgetype, ndim=4,  mode="c"] knowledge ):
        cdef int shape0 = img.shape[0]
        cdef int shape1 = img.shape[1]
        cdef int shape2 = img.shape[2]
        
        cdef int _nb_knowledge_layers = knowledge.shape[0]
        cdef int kshape0 = knowledge.shape[1]
        cdef int kshape1 = knowledge.shape[2]
        cdef int kshape2 = knowledge.shape[3]

        cdef np.ndarray[unsigned char, ndim=3,  mode="c"] seg = np.zeros( (shape0,
                                                                           shape1,
                                                                           shape2),
                                                                          dtype='uint8' )
        self.add_image_autocontext(img,seg,knowledge)

        cdef np.ndarray[pixeltype, ndim=4,  mode="c"] res = np.zeros( (_nb_knowledge_layers,
                                                                       kshape0,
                                                                       kshape1,
                                                                       kshape2),
                                                                      dtype='float32' )
        self.thisptr.debug_mean_knowledge( 0, <pixeltype*> res.data )
        
        return res    

        
    def grow(self, int nb_samples, int nb_background_samples ):
        self.thisptr.grow(nb_samples,nb_background_samples)
        self.nb_labels = self.thisptr.get_nb_labels()
        
    def predict_hard( self,
                      np.ndarray[pixeltype, ndim=3,  mode="c"] img,
                      np.ndarray[unsigned char, ndim=3,  mode="c"] mask ):
        cdef int shape0 = img.shape[0]
        cdef int shape1 = img.shape[1]
        cdef int shape2 = img.shape[2]
        cdef np.ndarray[unsigned char, ndim=3,  mode="c"] res = np.zeros( (shape0,
                                                                 shape1,
                                                                 shape2),
                                                                dtype='uint8' )
        self.thisptr.predict_hard( <pixeltype*> img.data,
                                    <unsigned char*> mask.data,
                                    shape0,
                                    shape1,
                                    shape2,
                                    <unsigned char*> res.data )
        return res
    
    def predict_soft( self,
                      np.ndarray[pixeltype, ndim=3,  mode="c"] img,
                      np.ndarray[unsigned char, ndim=3,  mode="c"] mask ):
        cdef int shape0 = img.shape[0]
        cdef int shape1 = img.shape[1]
        cdef int shape2 = img.shape[2]
        cdef np.ndarray[knowledgetype, ndim=4,  mode="c"] res = np.zeros( (self.nb_labels,
                                                                       shape0,
                                                                       shape1,
                                                                       shape2),
                                                                      dtype='float32' )
        self.thisptr.predict_soft( <pixeltype*> img.data,
                                    <unsigned char*> mask.data,
                                    shape0,
                                    shape1,
                                    shape2,
                                    self.nb_labels,
                                    <knowledgetype*> res.data )
        return res
    def predict_autocontext( self,
                             np.ndarray[pixeltype, ndim=3,  mode="c"] img,
                             np.ndarray[knowledgetype, ndim=4,  mode="c"] knowledge,
                             np.ndarray[unsigned char, ndim=3,  mode="c"] mask,
                             double ksampling,
                             np.ndarray[double, ndim=1,  mode="c"] metadata=np.zeros(0,dtype='float64') ):
        cdef int shape0 = img.shape[0]
        cdef int shape1 = img.shape[1]
        cdef int shape2 = img.shape[2]
        
        cdef int _nb_knowledge_layers = knowledge.shape[0]
        cdef int kshape0 = knowledge.shape[1]
        cdef int kshape1 = knowledge.shape[2]
        cdef int kshape2 = knowledge.shape[3]
        cdef vector[double] vect_metadata = metadata
        cdef np.ndarray[knowledgetype, ndim=4,  mode="c"] res = np.zeros( (self.nb_labels,
                                                                           shape0,
                                                                           shape1,
                                                                           shape2),
                                                                      dtype='float32' )

        self.thisptr.predict_autocontext( <pixeltype*> img.data,
                                           <unsigned char*> mask.data,
                                           shape0,
                                           shape1,
                                           shape2,
                                           <knowledgetype*> knowledge.data,
                                           _nb_knowledge_layers,
                                           kshape0,
                                           kshape1,
                                           kshape2,
                                           ksampling,
                                           vect_metadata,
                                           <knowledgetype*> res.data )
        return res
    
    def write(self, string folder):
        self.thisptr.write(folder)

    def get_feature_importance(self):
        return self.thisptr.get_feature_importance()




        
 

        
 
  
 
   
 
 
  

 
 
 
