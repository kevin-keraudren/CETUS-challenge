#include "Forest.h"
#include "SimpleTests.h"

class SimpleForest {
    Forest< std::vector<float> >* forest;
    bool verbose;

 public:
    SimpleForest( int ntrees,
                  double bagging=0.7,
                  int max_depth=10,
                  int min_items=20,
                  int nb_tests=1000,
                  int parallel=-1,
                  std::string test=std::string("axis"),
                  bool verbose=true ) {

        TreeParams< std::vector<float> >* params;
        Test< std::vector<float> >* test_class;
        
        if (test == "axis") {
            test_class = new AxisAligned();
        }
        else if (test == "linear") {
            test_class = new Linear();
        }
        else if (test == "linearND") {
            test_class = new LinearND();
        }
        else if (test == "parabola") {
            test_class = new Parabola();
        } 
        else {
            std::cout << "Unknown test type\n";
            exit(1);
        }

        params = new TreeParams< std::vector<float> >( max_depth,
                                                       min_items,
                                                       nb_tests,
                                                       test_class );

        params->set_bagging(bagging);
        params->set_ntrees(ntrees);
        params->set_parallel(-1);

        this->forest = new Forest< std::vector<float> >(params,0);
        this->verbose = verbose;
    }

    SimpleForest( std::string folder,
                  std::string test=std::string("axis") ) {
        Test< std::vector<float> >* test_class;
        if (test == "axis") {
            test_class = new AxisAligned();
        }
        else if (test == "linear") {
            test_class = new Linear();
        }
        else if (test == "linearND") {
            test_class = new LinearND();
        }        
        else if (test == "parabola") {
            test_class = new Parabola();
        } 
        else {
            std::cout << "Unknown test type\n";
            exit(1);
        }
        this->forest = new Forest< std::vector<float> >(folder,test_class);
    }    
    
    void grow_classification( std::vector< std::vector<float> >& points,
                              std::vector<int>& responses ) {
        this->forest->grow(points, responses, this->verbose);

    }
    void grow_regression( std::vector< std::vector<float> >& points,
                          std::vector< std::vector<double> >& responses ) {
        this->forest->grow(points, responses, this->verbose);

    }    

    std::vector<int> predict_hard( std::vector< std::vector<float> >& points ) {
        std::vector<int> predictions(points.size());
        for ( size_t i = 0; i < points.size(); i++ )
            predictions[i] = this->forest->predict_hard(points[i]);
        return predictions;
    }

    std::vector<std::vector<double> > predict_soft(
        std::vector<std::vector<float> >& points ) {
        std::vector<std::vector<double> > predictions(points.size());
        for ( size_t i = 0; i < points.size(); i++ )
            predictions[i] = this->forest->predict_soft(points[i]);
        return predictions;
    }

    std::vector< std::vector<double> > predict_regression( std::vector< std::vector<float> >& points ) {
        std::vector< std::vector<double> > predictions(points.size());
        for ( size_t i = 0; i < points.size(); i++ ) {
            predictions[i] = this->forest->predict_regression(points[i]);
        }
        return predictions;
    }

    void write( std::string folder ) {
        this->forest->write(folder);
    }

    void set_verbose( bool _verbose ) {
        this->verbose = _verbose;
    }    

    void set_bagging( double b) {
        this->forest->set_bagging(b);
    }
    void set_ntrees( int n ) {
        this->forest->set_ntrees(n);
    }
    void set_min_items( int n ) {
        this->forest->set_min_items(n);
    }
    void set_max_depth( int n ) {
        this->forest->set_max_depth(n);
    }
    void set_nb_tests( int n ) {
        this->forest->set_nb_tests(n);
    }     
    void set_test( std::string test ) {
        Test< std::vector<float> >* test_class;
        
        if (test == "axis") {
            test_class = new AxisAligned();
        }
        else if (test == "linear") {
            test_class = new Linear();
        }
        else if (test == "parabola") {
            test_class = new Parabola();
        } 
        else {
            std::cout << "Unknown test type\n";
            exit(1);
        }
        
        this->forest->set_test(test_class);
    }     
};
