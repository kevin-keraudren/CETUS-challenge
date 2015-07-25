#include "Forest.h"
#include "IntegralTest.h"

inline size_t index( size_t i, size_t j, size_t k,
                     size_t shape0, size_t shape1, size_t shape2 ) {
    return k + shape2*( j + shape1*i );
}

inline size_t index( size_t i, size_t j, size_t k, size_t l,
                     size_t shape0, size_t shape1, size_t shape2, size_t shape3 ) {
    return l + shape3*( k + shape2*( j + shape1*i ) );
}

class ParallelPredictAutocontext;

void integral( pixeltype* img,
               int shape0,
               int shape1,
               int shape2,
               pixeltype* sum ) {
    // "Efficient Visual Event Detection using Volumetric Feature"
    pixeltype s;
    for ( int z = 0; z < shape0; z++ )
        for (int y = 0; y < shape1; y++ )
            for ( int x = 0; x < shape2; x++ ) {
                s = img[index(z,y,x,shape0,shape1,shape2)];
                if (z>0)
                    s += sum[index(z-1,y,x,shape0,shape1,shape2)];
                if (y>0)
                    s += sum[index(z,y-1,x,shape0,shape1,shape2)];
                if (x>0)
                    s += sum[index(z,y,x-1,shape0,shape1,shape2)];
                if (z>0 && y>0)
                    s -= sum[index(z-1,y-1,x,shape0,shape1,shape2)];
                if (z>0 && x>0)
                    s -= sum[index(z-1,y,x-1,shape0,shape1,shape2)];
                if (y>0 && x>0)
                    s -= sum[index(z,y-1,x-1,shape0,shape1,shape2)];
                if (z>0 && y>0 && x>0)
                    s += sum[index(z-1,y-1,x-1,shape0,shape1,shape2)];
                sum[index(z,y,x,shape0,shape1,shape2)] = s;
            }
}

// integral knowledge
void integral( knowledgetype* knowledge,
               int _nb_knowledge_layers,
               int shape0,
               int shape1,
               int shape2,
               knowledgetype* sum_knowledge ) {
    // "Efficient Visual Event Detection using Volumetric Feature"
    knowledgetype s;
    for ( int k = 0; k < _nb_knowledge_layers; k++ )
        for ( int z = 0; z < shape0; z++ )
            for (int y = 0; y < shape1; y++ )
                for ( int x = 0; x < shape2; x++ ) {
                    s = knowledge[index(k,z,y,x,_nb_knowledge_layers,shape0,shape1,shape2)];
                    if (z>0)
                        s += sum_knowledge[index(k,z-1,y,x,_nb_knowledge_layers,shape0,shape1,shape2)];
                    if (y>0)
                        s += sum_knowledge[index(k,z,y-1,x,_nb_knowledge_layers,shape0,shape1,shape2)];
                    if (x>0)
                        s += sum_knowledge[index(k,z,y,x-1,_nb_knowledge_layers,shape0,shape1,shape2)];
                    if (z>0 && y>0)
                        s -= sum_knowledge[index(k,z-1,y-1,x,_nb_knowledge_layers,shape0,shape1,shape2)];
                    if (z>0 && x>0)
                        s -= sum_knowledge[index(k,z-1,y,x-1,_nb_knowledge_layers,shape0,shape1,shape2)];
                    if (y>0 && x>0)
                        s -= sum_knowledge[index(k,z,y-1,x-1,_nb_knowledge_layers,shape0,shape1,shape2)];
                    if (z>0 && y>0 && x>0)
                        s += sum_knowledge[index(k,z-1,y-1,x-1,_nb_knowledge_layers,shape0,shape1,shape2)];
                    sum_knowledge[index(k,z,y,x,_nb_knowledge_layers,shape0,shape1,shape2)] = s;
                }
}

class IntegralForest {
    Forest< SlidingWindow >* forest;
    cv::RNG rng; // OpenCV random number generator
    std::vector< pixeltype* > images;
    std::vector< int* > shapes;
    std::vector< int* > kshapes;
    std::vector< unsigned char* > segmentations;
    std::vector< knowledgetype* > all_knowledge;
    std::vector< std::vector<double> > all_metadata;
    int nb_labels;
    int nb_knowledge_layers;
    int parallel;
    double ksampling;
    bool verbose;
    
    friend class ParallelPredictAutocontext;
    
 public:
    IntegralForest( int ntrees,
                    double bagging=0.7,
                    int max_depth=10,
                    int min_items=20,
                    int nb_tests=1000,
                    int _parallel=-1,
                    std::string test=std::string("block"),
                    bool verbose=true,
                    int cx=50, int cy=50, int cz=50,
                    int dx=50, int dy=50, int dz=50,
                    int nb_labels=2,
                    int nb_knowledge_layers=2,
                    double ksampling=1.0,
                    int nfeatures=1 ) {
        
        TreeParams< SlidingWindow >* params;

        if (test == "block") {
            BlockTest* test_class = new BlockTest();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }
        else if (test == "autocontext") {
            AutoContext* test_class = new AutoContext();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            test_class->set_nb_knowledge_layers( nb_knowledge_layers );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }
        else if (test == "autocontext2") {
            AutoContext2* test_class = new AutoContext2();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            test_class->set_nb_knowledge_layers( nb_knowledge_layers );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }
        else if (test == "autocontextN") {
            AutoContextN* test_class = new AutoContextN();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            test_class->set_nb_knowledge_layers( nb_knowledge_layers );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }         
        else if (test == "autocontextMetadata") {
            AutoContextMetadata* test_class = new AutoContextMetadata();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            test_class->set_nb_knowledge_layers( nb_knowledge_layers );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }
        else if (test == "adaptiveAutocontext") {
            AdaptiveAutoContext* test_class = new AdaptiveAutoContext();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            test_class->set_nb_knowledge_layers( nb_knowledge_layers );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }
        else if (test == "block2DAutocontext") {
            Block2DAutoContext* test_class = new Block2DAutoContext();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            test_class->set_nb_knowledge_layers( nb_knowledge_layers );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }
        else if (test == "autocontextDistancePrior") {
            AutoContextDistancePrior* test_class = new AutoContextDistancePrior();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            test_class->set_nb_knowledge_layers( nb_knowledge_layers );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }
        else if (test == "autocontextGradient") {
            AutoContextGradient* test_class = new AutoContextGradient();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            test_class->set_nb_knowledge_layers( nb_knowledge_layers );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }
        else if (test == "autocontextGradientDistancePrior") {
            AutoContextGradientDistancePrior* test_class = new AutoContextGradientDistancePrior();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            test_class->set_nb_knowledge_layers( nb_knowledge_layers );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }
        else if (test == "heartautocontext") {
            HeartAutoContext* test_class = new HeartAutoContext();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            test_class->set_nb_knowledge_layers( nb_knowledge_layers );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }
        else if (test == "patch") {
            PatchTest* test_class = new PatchTest();
            test_class->set_size( cx, cy, cz,
                                  dx, dy, dz );
            params = new TreeParams< SlidingWindow >( max_depth,
                                                      min_items,
                                                      nb_tests,
                                                      test_class );
        }        
        else {
            std::cout << "Unknown test type\n";
            std::cout << "Please choose between: block, autocontext or patch\n";
            exit(1);
        }

        params->set_bagging(bagging);
        params->set_ntrees(ntrees);
        params->set_parallel(_parallel);
        params->set_nfeatures(nfeatures);
        
        this->parallel = _parallel;

        this->forest = new Forest< SlidingWindow >(params,0);
        this->nb_labels = nb_labels;
        this->nb_knowledge_layers = nb_knowledge_layers;
        this->ksampling = ksampling;
        this->verbose = verbose;
    }

    IntegralForest( std::string folder,
                    std::string test=std::string("block"),
                    int _parallel=-1 ) {
        Test< SlidingWindow >* test_class;
        if (test == "block")
            test_class = new BlockTest();
        else if (test == "autocontext")
            test_class = new AutoContext();
        else if (test == "autocontext2")
            test_class = new AutoContext2();
        else if (test == "autocontextN")
            test_class = new AutoContextN();         
        else if (test == "autocontextMetadata")
            test_class = new AutoContextMetadata();
        else if (test == "adaptiveAutocontext")
            test_class = new AdaptiveAutoContext();
        else if (test == "block2DAutocontext")
            test_class = new Block2DAutoContext();
        else if (test == "autocontextDistancePrior")
            test_class = new AutoContextDistancePrior();
        else if (test == "autocontextGradient")
            test_class = new AutoContextGradient();
        else if (test == "autocontextGradientDistancePrior")
            test_class = new AutoContextGradientDistancePrior();
        else if (test == "heartautocontext")
            test_class = new HeartAutoContext();
        else if (test == "patch")
            test_class = new PatchTest();
        else {
            std::cout << "Unknown test type\n";
            exit(1);
        }

        this->parallel = _parallel;

        this->forest = new Forest< SlidingWindow >( folder,
                                                    test_class,
                                                    _parallel );
    }

    ~IntegralForest() {

        for ( int i = 0; i < images.size(); i++ ) {
            delete images[i];
            delete segmentations[i];
            delete shapes[i];
        }
        for ( int i = 0; i < all_knowledge.size(); i++ ) {
            delete all_knowledge[i];
            delete kshapes[i];
        }
        
        delete forest;
    }

    int get_nb_labels() { return this->forest->get_nb_labels(); }
    int get_nb_knowledge_layers() { return this->get_nb_knowledge_layers(); }
    
    void add_image( pixeltype* img,
                    int shape0,
                    int shape1,
                    int shape2,
                    unsigned char* _seg ) {
        int* shape = new int[3];
        shape[0] = shape0;
        shape[1] = shape1;
        shape[2] = shape2;
        this->shapes.push_back(shape);
        unsigned char* seg = new unsigned char[shape0*shape1*shape2];
        for (size_t i = 0; i < shape0*shape1*shape2; i++)
            seg[i] = _seg[i];
        this->segmentations.push_back(seg);
        pixeltype* sum = new pixeltype[shape0*shape1*shape2];
        integral( img, shape0, shape1, shape2, sum );
        this->images.push_back(sum);
    }

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
                                std::vector<double> metadata ) {
        assert(_nb_knowledge_layers==this->nb_knowledge_layers);
        
        int* shape = new int[3];
        shape[0] = shape0;
        shape[1] = shape1;
        shape[2] = shape2;
        this->shapes.push_back(shape);
        unsigned char* seg = new unsigned char[shape0*shape1*shape2];
        for (size_t i = 0; i < shape0*shape1*shape2; i++)
            seg[i] = _seg[i];
        this->segmentations.push_back(seg);
        pixeltype* sum = new pixeltype[shape0*shape1*shape2];
        integral( img, shape0, shape1, shape2, sum );
        this->images.push_back(sum);

        int* kshape = new int[3];
        kshape[0] = kshape0;
        kshape[1] = kshape1;
        kshape[2] = kshape2;
        this->kshapes.push_back(kshape);
        knowledgetype* sum_knowledge = new knowledgetype[_nb_knowledge_layers*kshape0*kshape1*kshape2];
        integral( knowledge, _nb_knowledge_layers, kshape0, kshape1, kshape2, sum_knowledge );
        this->all_knowledge.push_back(sum_knowledge);

        this->all_metadata.push_back(metadata);
    }
 
    void grow( int nb_samples,
               int nb_background_samples,
               int seed=time(NULL) ) {

        rng = cv::RNG(seed); // initialize random number generator
        
        std::vector< SlidingWindow > points;
        std::vector< int > responses;

        int current_nb_samples;
        for ( int i = 0; i < this->images.size(); i++ ) {
            pixeltype* img = this->images[i];
            unsigned char* seg = this->segmentations[i];
            int shape0 = this->shapes[i][0];
            int shape1 = this->shapes[i][1];
            int shape2 = this->shapes[i][2];
            for ( int cl = 0; cl < this->nb_labels; cl++ ) {
                std::vector<int> X;
                std::vector<int> Y;
                std::vector<int> Z;
                for ( int z = 0; z < shape0; z++ )
                    for ( int y = 0; y < shape1; y++ )
                        for ( int x = 0; x < shape2; x++ ) {
                            if (seg[index(z,y,x,shape0,shape1,shape2)]==cl){
                                X.push_back(x);
                                Y.push_back(y);
                                Z.push_back(z);
                            }
                        }
                if (cl==0)
                    current_nb_samples = nb_background_samples;
                else
                    current_nb_samples = nb_samples;
                for ( int n = 0; n < current_nb_samples; n++ ) {
                    int s = rng.uniform(0,X.size());
                    if (this->all_knowledge.size() == 0)
                        points.push_back( SlidingWindow( img,
                                                         shape0,
                                                         shape1,
                                                         shape2,
                                                         X[s],
                                                         Y[s],
                                                         Z[s] ) );
                    else
                        points.push_back( SlidingWindow( img,
                                                         shape0,
                                                         shape1,
                                                         shape2,
                                                         this->all_knowledge[i],
                                                         this->nb_knowledge_layers,
                                                         this->kshapes[i][0],
                                                         this->kshapes[i][1],
                                                         this->kshapes[i][2],
                                                         this->ksampling,
                                                         X[s],
                                                         Y[s],
                                                         Z[s],
                                                         &(this->all_metadata[i]) ) );
                    responses.push_back(cl);
                }
            }
        }
                
        this->forest->grow(points, responses, this->verbose);

    }

    std::vector<double> get_feature_importance() {
        return this->forest->get_feature_importance();
    }

    void predict_hard( pixeltype* img,
                       unsigned char* mask,
                       int shape0,
                       int shape1,
                       int shape2,
                       unsigned char* res ) {
        
        pixeltype* sum = new pixeltype[shape0*shape1*shape2];
        integral( img, shape0, shape1, shape2, sum );
        
        SlidingWindow point( sum,
                             shape0,
                             shape1,
                             shape2,
                             0, 0, 0);

        size_t idx = 0;
        for ( int z = 0; z < shape0; z++ )
            for ( int y = 0; y < shape1; y++ )
                for ( int x = 0; x < shape2; x++ ) {
                    if (mask[idx]>0) {
                        point.set(x,y,z);
                        res[idx] = this->forest->predict_hard(point);
                    }
                    idx++;
                }

        delete sum;
    }

    void predict_soft( pixeltype* img,
                       unsigned char* mask,
                       int shape0,
                       int shape1,
                       int shape2,
                       int _nb_labels,
                       knowledgetype* res ) {

        pixeltype* sum = new pixeltype[shape0*shape1*shape2];
        integral( img, shape0, shape1, shape2, sum );
        
        SlidingWindow point( sum,
                             shape0,
                             shape1,
                             shape2,
                             0, 0, 0);

        std::vector<double> prediction;
        for ( int z = 0; z < shape0; z++ )
            for ( int y = 0; y < shape1; y++ )
                for ( int x = 0; x < shape2; x++ ) {
                    if (mask[index(z,y,x,shape0,shape1,shape2)] == 0)
                        continue;
                    point.set(x,y,z);
                    prediction = this->forest->predict_soft(point);
                    for ( int k = 0; k < prediction.size(); k++ )
                        res[index(k,z,y,x,_nb_labels,shape0,shape1,shape2)] = prediction[k];
                }
        delete sum;
    }

    void predict_autocontext(  pixeltype* img,
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
                               std::vector<double> metadata, 
                               knowledgetype* res );
    
    void write( std::string folder ) {
        this->forest->write(folder);
    }

    void debug_mean( int n,
                     pixeltype* res ) {
        int shape0 = this->shapes[n][0];
        int shape1 = this->shapes[n][1];
        int shape2 = this->shapes[n][2];

        SlidingWindow point( this->images[n],
                             shape0,
                             shape1,
                             shape2,
                             0, 0, 0);
        for ( int z = 0; z < shape0; z++ )
            for ( int y = 0; y < shape1; y++ )
                for ( int x = 0; x < shape2; x++ ) {
                    point.set(x,y,z);
                    // std::cout << point.mean(0,0,0,10,10,10) <<"\n";
                    res[index(z,y,x,shape0,shape1,shape2)] = point.mean(0,0,0,5,5,5);
                }
    }

    void debug_mean_knowledge( int n,
                               pixeltype* res ) {
        int shape0 = this->shapes[n][0];
        int shape1 = this->shapes[n][1];
        int shape2 = this->shapes[n][2];

        int kshape0 = this->kshapes[n][0];
        int kshape1 = this->kshapes[n][1];
        int kshape2 = this->kshapes[n][2];        

        SlidingWindow point( this->images[n],
                             shape0,
                             shape1,
                             shape2,
                             this->all_knowledge[n],
                             this->nb_knowledge_layers,
                             kshape0,
                             kshape1,
                             kshape2,
                             this->ksampling,
                             0, 0, 0,
                             &(this->all_metadata[n]) );
        for ( int k = 0; k < this->nb_knowledge_layers; k++ )
            for ( int z = 0; z < kshape0; z++ )
                for ( int y = 0; y < kshape1; y++ )
                    for ( int x = 0; x < kshape2; x++ ) {
                        point.set( (int)(this->ksampling*x),
                                   (int)(this->ksampling*y),
                                   (int)(this->ksampling*z) );
                        // std::cout << point.mean(0,0,0,10,10,10) <<"\n";
                        res[index(k,z,y,x,nb_knowledge_layers,kshape0,kshape1,kshape2)] = point.mean_knowledge(0,0,0,5,5,5,k);
                    }
    }    

    void debug_data() {}

};

class ParallelPredictAutocontext {
    IntegralForest* integralForest;
    pixeltype* sum;
    knowledgetype* sum_knowledge;
    unsigned char* mask;
    knowledgetype* res;
    std::vector<double>* metadata;
    int shape0, shape1, shape2;
    int kshape0, kshape1, kshape2;
    int nb_knowledge_layers;
    double ksampling;
    int nb_labels;

public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t z = r.begin(); z != r.end(); z++ ) {

        SlidingWindow point( sum,
                             shape0,
                             shape1,
                             shape2,
                             sum_knowledge,
                             nb_knowledge_layers,
                             kshape0,
                             kshape1,
                             kshape2,
                             ksampling,
                             0, 0, 0,
                             metadata );
         std::vector<double> prediction;
         for ( int y = 0; y < shape1; y++ )
             for ( int x = 0; x < shape2; x++ ) {
                 if (mask[index(z,y,x,shape0,shape1,shape2)]==0)
                     continue;
                 point.set(x,y,z);
                 prediction = integralForest->forest->predict_soft(point);
                 for ( int k = 0; k < prediction.size(); k++ )
                     res[index(k,z,y,x,nb_labels,shape0,shape1,shape2)] = prediction[k];
                }
                
    }
  }
 ParallelPredictAutocontext( IntegralForest* _integralForest,
                             pixeltype* _sum,
                             unsigned char* _mask,
                             knowledgetype* _res,
                             int _shape0,
                             int _shape1,
                             int _shape2,
                             knowledgetype* _sum_knowledge,
                             int _nb_knowledge_layers,
                             int _kshape0,
                             int _kshape1,
                             int _kshape2,
                             double _ksampling,
                             std::vector<double>* _metadata,
                             int _nb_labels )
    {
        integralForest = _integralForest;
        sum = _sum;
        mask = _mask;
        res = _res;
        shape0 = _shape0;
        shape1 = _shape1;
        shape2 = _shape2;        
        sum_knowledge = _sum_knowledge;
        nb_knowledge_layers = _nb_knowledge_layers;
        kshape0 = _kshape0;
        kshape1 = _kshape1;
        kshape2 = _kshape2;
        ksampling = _ksampling;
        nb_labels = _nb_labels;
        metadata = _metadata;
    }
};

void IntegralForest::predict_autocontext( pixeltype* img,
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
                                          std::vector<double> metadata,
                                          knowledgetype* res ) {

        pixeltype* sum = new pixeltype[shape0*shape1*shape2];
        integral( img, shape0, shape1, shape2, sum );
        knowledgetype* sum_knowledge = new knowledgetype[_nb_knowledge_layers*shape0*shape1*shape2];
        integral( knowledge, _nb_knowledge_layers, kshape0, kshape1, kshape2, sum_knowledge );
        
        SlidingWindow point( sum,
                             shape0,
                             shape1,
                             shape2,
                             sum_knowledge,
                             _nb_knowledge_layers,
                             kshape0,
                             kshape1,
                             kshape2,
                             ksampling,
                             0, 0, 0,
                             &metadata );

        /* std::vector<double> prediction; */
        /* for ( int z = 0; z < shape0; z++ ) */
        /*     for ( int y = 0; y < shape1; y++ ) */
        /*         for ( int x = 0; x < shape2; x++ ) { */
        /*             if (mask[index(z,y,x,shape0,shape1,shape2)]==0) */
        /*                 continue; */
        /*             point.set(x,y,z); */
        /*             prediction = this->forest->predict_soft(point); */
        /*             for ( int k = 0; k < prediction.size(); k++ ) */
        /*                res[index(k,z,y,x,_nb_labels,shape0,shape1,shape2)] = prediction[k]; */
        /*         } */

        // Setting maximum number of threads
        tbb::task_scheduler_init init( this->parallel );
        tbb::parallel_for(tbb::blocked_range<size_t>(0, shape0 ),
                          ParallelPredictAutocontext( this,
                                                      sum,
                                                      mask,
                                                      res,
                                                      shape0,
                                                      shape1,
                                                      shape2,
                                                      sum_knowledge,
                                                      _nb_knowledge_layers,
                                                      kshape0,
                                                      kshape1,
                                                      kshape2,
                                                      ksampling,
                                                      &metadata,
                                                      nb_labels )
                          );

        delete sum;
        delete sum_knowledge;
            
    }
