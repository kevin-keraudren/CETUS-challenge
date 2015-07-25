#ifndef __FOREST_H__
#define __FOREST_H__

#include <vector>
#include "Tree.h"
#include "utils.h"

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>        // cv::Mat etc, always need this
#include <opencv2/imgproc/imgproc.hpp>  // all the image processing functions
#include <opencv2/highgui/highgui.hpp>  // Display and file I/O

// Classification parallel
template <class PointType>
class GrowClassificationTrees {
    std::vector< Tree<PointType> >& trees;
    TreeParams<PointType>* tree_params;
    std::vector<PointType>& points;
    std::vector<int>& responses;
    std::vector< std::vector<int> >& selection;
    bool verbose;

public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t i = r.begin(); i != r.end(); ++i ) {

        trees[i] = Tree< PointType >(tree_params, time(NULL) + i*1000);
        std::vector<PointType> selected_points;
        std::vector<int> selected_responses;
        for ( int n = 0; n < selection[i].size(); n++ ) {
            selected_points.push_back( points[selection[i][n]] );
            selected_responses.push_back( responses[selection[i][n]] );
        }
        trees[i].grow( selected_points, selected_responses, verbose );
    }
  }
 GrowClassificationTrees( std::vector< Tree<PointType> >& _trees,
                          TreeParams<PointType>* _tree_params,
                          std::vector<PointType>& _points,
                          std::vector<int>& _responses,
                          std::vector< std::vector<int> >& _selection,
                          bool _verbose ) :
    trees(_trees), tree_params(_tree_params),
        points(_points), responses(_responses),
        selection(_selection), verbose(_verbose)
    { }
};

// Regression parallel
template <class PointType>
class GrowRegressionTrees {
    std::vector< Tree<PointType> >& trees;
    TreeParams<PointType>* tree_params;
    std::vector<PointType>& points;
    std::vector< std::vector<double> >& responses;
    std::vector< std::vector<int> >& selection;
    bool verbose;

public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t i = r.begin(); i != r.end(); ++i ) {

        trees[i] = Tree< PointType >(tree_params, time(NULL) + i*10000);
        std::vector<PointType> selected_points;
        std::vector< std::vector<double> > selected_responses;
        for ( int n = 0; n < selection[i].size(); n++ ) {
            selected_points.push_back( points[selection[i][n]] );
            selected_responses.push_back( responses[selection[i][n]] );
        }
        trees[i].grow( selected_points, selected_responses, verbose );
    }
  }
 GrowRegressionTrees( std::vector< Tree<PointType> >& _trees,
                      TreeParams<PointType>* _tree_params,
                      std::vector<PointType>& _points,
                      std::vector< std::vector<double> >& _responses,
                      std::vector< std::vector<int> >& _selection,
                      bool _verbose ) :
    trees(_trees), tree_params(_tree_params),
        points(_points), responses(_responses),
        selection(_selection)
    {
        verbose = _verbose;
    }
};

template <class PointType>
class PredictTrees {
    std::vector< Tree<PointType> >& trees;
    PointType& point;
    std::vector<double>& predictions;

public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t i = r.begin(); i != r.end(); ++i ) {
        std::vector<double> p = trees[i].predict(point);
        for ( int c = 0; c < trees[i].get_nb_labels(); c++ )
            predictions[c] += p[c];
    }
  }
 PredictTrees( std::vector< Tree<PointType> >& _trees,
               PointType& _point,
               std::vector<double>& _predictions ) :
    trees(_trees), point(_point), predictions(_predictions)
    { }
};

template <class PointType>
class ParallelRead {
    std::vector< Tree<PointType> >& trees;
    std::vector<std::string>& tree_files;
    Test<PointType>* test_class;
    
public:
    void operator() ( const tbb::blocked_range<size_t>& r ) const {
    for ( size_t i = r.begin(); i != r.end(); ++i ) {
        trees[i] = Tree< PointType >( tree_files[i], test_class );
    }
  }
 ParallelRead( std::vector< Tree<PointType> >& _trees,
               std::vector<std::string>& _tree_files,
               Test<PointType>* _test_class ) :
    trees(_trees), tree_files(_tree_files)
    {
        test_class = _test_class;
    }
};

void recursive_delete( Node** n ) {
    if ( *n != NULL ) {
        if (! (*n)->leaf ) {
            recursive_delete(&((*n)->left));
            recursive_delete(&((*n)->right));
        }
        delete *n;
        n = NULL;
    }
}

/*************************** Forest *********************************/

template <class PointType>
class Forest {
    TreeParams<PointType>* params; // tree parameters
    int nb_labels; // number of labels: we assume labels are always consecutive
                   // integers starting from zero
    std::vector< Tree<PointType> > trees;
    uint64 rng_state;
    double oob;
    double oob_min;
    double oob_max;
    
 public:

 Forest( TreeParams<PointType>* _params, uint64 _rng_state ):
    params(_params), rng_state(_rng_state)
    {
        this->nb_labels = 0;
    }

    ~Forest() {
        //http://stackoverflow.com/questions/16198029/destructor-gets-called-before-the-end-of-scope
        for ( int i = 0; i < this->size(); i++ )
             recursive_delete( &(this->trees[i].root) );
    };

    int size() { return this->trees.size(); }
    int get_nb_labels() { return this->nb_labels; }
    double get_oob() { return this->oob; }
    double get_oob_min() { return this->oob_min; }
    double get_oob_max() { return this->oob_max; }
    void set_modulo( double mod ) { this->params->set_modulo( mod ); }
    void set_bagging( double b ) { this->params->set_bagging( b ); }
    void set_ntrees( int n ) { this->params->set_ntrees( n ); }
    void set_nb_tests( int n ) { this->params->set_nb_tests( n ); }
    void set_test( Test<PointType>* test_class ) { this->params->set_test( test_class ); }  
    void set_min_items(int n) { this->params->set_min_items( n ); }
    void set_max_depth(int n) { this->params->set_max_depth( n ); }
    
    int prepare_bagging( int nb_points,
                          std::vector< std::vector<int> >& selection,
                         std::vector< std::vector<int> >& not_selected,
                         std::vector<bool>& used_in_oob ) {
        selection.resize( this->params->ntrees );
        not_selected.resize( this->params->ntrees );
        used_in_oob.resize( nb_points, false );

        // No bagging (regression?)
        if ( this->params->bagging <= 0 ) {
        for ( int i = 0; i < this->params->ntrees; i++ )
            for ( size_t n = 0; n < nb_points; n++ )
                selection[i].push_back( n );
            return 0;
        }
        
        cv::RNG rng(this->rng_state);
        for ( int i = 0; i < this->params->ntrees; i++ )
            for ( size_t n = 0; n < nb_points*this->params->bagging; n++ ) {
                int p = rng.uniform(0,nb_points);
                selection[i].push_back( p );
            }

        for ( int i = 0; i < this->params->ntrees; i++ ) {
            not_selected[i].resize( nb_points, 0 );
            for ( size_t n = 0; n < selection[i].size(); n++ )
                not_selected[i][selection[i][n]] = 1;
        }

        int count = 0;
        for ( int n = 0; n < nb_points; n++ ) {
            bool seen = true;
            for ( int i = 0; i < this->params->ntrees; i++ )
                if ( not_selected[i][n] == 0 ) {
                    seen = false;
                    break;
                }
            if ( ! seen ) {
                count++;
                used_in_oob[n] = true;
            }
        }

        return count;
    }

    int balanced_bagging( std::vector<int>& responses,
                          std::vector< std::vector<int> >& selection,
                          std::vector< std::vector<int> >& not_selected,
                          std::vector<bool>& used_in_oob ) {
        selection.resize( this->params->ntrees );
        not_selected.resize( this->params->ntrees );
        used_in_oob.resize( responses.size(), false );

        // No bagging (regression?)
        if ( this->params->bagging <= 0 ) {
        for ( int i = 0; i < this->params->ntrees; i++ )
            for ( size_t n = 0; n < responses.size(); n++ )
                selection[i].push_back( n );
            return 0;
        }        

        // Balanced Random Forests
        // http://www.stat.berkeley.edu/tech-reports/666.pdf
        std::vector< std::vector<int> > classes(this->nb_labels);
        for ( size_t i = 0; i < responses.size(); i++ )
            classes[responses[i]].push_back(i);

        int small_class = 0;        
        for ( int c = 0; c < this->nb_labels; c++ )
            if ( classes[c].size() > small_class )
                small_class = classes[c].size();

        cv::RNG rng(this->rng_state);
        for ( int i = 0; i < this->params->ntrees; i++ )
            for ( int c = 0; c < this->nb_labels; c++ )
                for ( size_t n = 0; n < small_class*this->params->bagging; n++ ) {
                    int p = rng.uniform(0,classes[c].size());
                    selection[i].push_back( classes[c][p] );
                }

        for ( int i = 0; i < this->params->ntrees; i++ ) {
            not_selected[i].resize( responses.size(), 0 );
            for ( size_t n = 0; n < selection[i].size(); n++ )
                not_selected[i][selection[i][n]] = 1;
        }

        int count = 0;
        for ( int n = 0; n < responses.size(); n++ ) {
            bool seen = true;
            for ( int i = 0; i < this->params->ntrees; i++ )
                if ( not_selected[i][n] == 0 ) {
                    seen = false;
                    break;
                }
            if ( ! seen ) {
                count++;
                used_in_oob[n] = true;
            }
        }
        return count;
    }

    // Classification
    void grow( std::vector<PointType>& points,
               std::vector<int>& responses,
               bool verbose ) {

        // set number of labels
        int max_label = -1;
        for ( int i = 0; i < responses.size(); i++ )
            if ( responses[i] > max_label )
                max_label = responses[i];
        this->nb_labels = max_label + 1;

        // Bagging
        std::vector< std::vector<int> > selection;
        std::vector< std::vector<int> > not_selected;
        std::vector<bool> used_in_oob;
        int count_for_oob =
            this->prepare_bagging( points.size(), selection, not_selected, used_in_oob );

        /* int count_for_oob = */
        /*     this->balanced_bagging( responses, selection, not_selected, used_in_oob ); */

        this->trees.resize( this->params->ntrees, Tree< PointType >(NULL, 0) );
        
        if ( this->params->parallel != 0 ) {
            // Setting maximum number of threads
            tbb::task_scheduler_init init( this->params->parallel );
            tbb::parallel_for(tbb::blocked_range<size_t>(0, this->params->ntrees ),
                              GrowClassificationTrees<PointType>( this->trees,
                                                                  this->params,
                                                                  points,
                                                                  responses,
                                                                  selection,
                                                                  verbose )
                              );
        }
        else {       
            for ( int i = 0; i < this->params->ntrees; i++ ) {
                this->trees[i] = Tree< PointType >(this->params, time(NULL) + i*1000);
                std::vector<PointType> selected_points;
                std::vector<int> selected_responses;
                for ( int n = 0; n < selection[i].size(); n++ ) {
                    selected_points.push_back( points[selection[i][n]] );
                    selected_responses.push_back( responses[selection[i][n]] );
                }
                this->trees[i].grow( selected_points, selected_responses, verbose );
            }
        }

        // Out-of-bag (oob) error estimate
        // http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr
        std::vector< std::vector<double> > classification;
        classification.resize( points.size() );
        for ( int n = 0; n < points.size(); n++ )
            classification[n].resize( this->nb_labels, 0 );
        for ( int i = 0; i < this->params->ntrees; i++ )
            for ( int n = 0; n < points.size(); n++ ) {
                if ( not_selected[i][n] == 0 ) {
                    std::vector<double> p = this->trees[i].predict( points[n] );
                    for ( int c = 0; c < this->nb_labels; c++ )
                        classification[n][c] += p[c];
                }
            }

        this->oob = 0;
        for ( int n = 0; n < points.size(); n++ ) {
            if ( ! used_in_oob[n] )
                continue;
            int response = -1;
            double max_count = -1;
            for ( int c = 0; c < this->nb_labels; c++ )
                if ( classification[n][c] > max_count ) {
                response = c;
                max_count = classification[n][c];
                }
            if ( response != responses[n] )
                this->oob++;
        }

        if ( count_for_oob > 0 )
            this->oob /= count_for_oob;

        if ( verbose ) {
            std::cout << "Out-of-bag (OOB) error estimate:" << std::endl;
            std::cout << "Number of samples used in OOB: " << count_for_oob << std::endl;
            std::cout << "OOB (misclassification): " << this->oob << std::endl;
        }

    }

    // Regression
    void grow( std::vector<PointType>& points,
               std::vector< std::vector<double> >& responses,
               bool verbose ) {
        
        // Bagging
        std::vector< std::vector<int> > selection;
        std::vector< std::vector<int> > not_selected;
        std::vector<bool> used_in_oob;
        int count_for_oob =
            this->prepare_bagging( points.size(), selection, not_selected, used_in_oob );

        this->trees.resize( this->params->ntrees, Tree< PointType >( NULL, 0) );
        
        if ( this->params->parallel != 0 ) {
            // Setting maximum number of threads
            tbb::task_scheduler_init init( this->params->parallel );
            tbb::parallel_for(tbb::blocked_range<size_t>(0, this->params->ntrees ),
                              GrowRegressionTrees<PointType>( this->trees,
                                                              this->params,
                                                              points,
                                                              responses,
                                                              selection,
                                                              verbose )
                              );
        }
        else {
            for ( int i = 0; i < this->params->ntrees; i++ ) {
                this->trees[i] = Tree< PointType >(this->params, i*10000);
                std::vector<PointType> selected_points;
                std::vector< std::vector<double> > selected_responses;
                for ( int n = 0; n < selection[i].size(); n++ ) {
                    selected_points.push_back( points[selection[i][n]] );
                    selected_responses.push_back( responses[selection[i][n]] );
                }
                this->trees[i].grow( selected_points, selected_responses, verbose );
            }
        }

        // Out-of-bag (oob) error estimate
        // http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr
        if ( count_for_oob == 0 ) {
            this->oob = -1;
            return;
        }
        std::vector< std::vector<double> > regression;
        regression.resize( points.size() );
        int dim = responses[0].size();
        for ( int n = 0; n < points.size(); n++ )
            regression[n].resize( dim, 0 );
        for ( int n = 0; n < points.size(); n++ ) {
            int count = 0;
            for ( int i = 0; i < this->params->ntrees; i++ ) {
                if ( not_selected[i][n] == 0 ) {
                    count++;
                    std::vector<double> r = this->trees[i].predict( points[n] );
                    for ( int d = 0; d < dim; d++ )
                        regression[n][d] += r[d];
                }
            }
            for ( int d = 0; d < dim; d++ )
                regression[n][d] /= count;
        }

        this->oob = 0;
        if ( this->params->modulo == -1 ) {        
            for ( int n = 0; n < points.size(); n++ ) {
                if ( ! used_in_oob[n] )
                    continue;
                for ( int d = 0; d < dim; d++ ) {
                    double tmp = regression[n][d] - responses[n][d];
                    this->oob += tmp*tmp;
                }
            }

            if ( count_for_oob > 0 )
                this->oob /= count_for_oob;
            this->oob = sqrt( this->oob );

            if ( verbose ) {
                std::cout << "Out-of-bag (OOB) error estimate:" << std::endl;
                std::cout << "Number os samples used in OOB: " << count_for_oob << std::endl;
                std::cout << "OOB (Root mean-square error): " << this->oob << std::endl;
            }
        }
        else {
            std::vector<double> errors;
            cv::RNG rng;
            for ( int n = 0; n < points.size(); n++ ) {
                if ( ! used_in_oob[n] )
                    continue;
                double norm = sqrt( regression[n][0]*regression[n][0] +
                               regression[n][1]*regression[n][1] );
                double rnorm = sqrt( responses[n][0]*responses[n][0] +
                               responses[n][1]*responses[n][1] );
                double theta;
                double response;
                if ( norm == 0 ) {
                    // Random guess
                    theta = rng.uniform( 0.0, this->params->modulo );
                    std::cout << "ZERO NORM\n";
                    exit(1);
                }
                theta = acos( regression[n][0] / norm );
                theta *= sign( regression[n][1] );
                response = acos( responses[n][0] / rnorm );
                response *= sign( responses[n][1] );
                double err = acos(cos(theta - response))/(2*CV_PI)*this->params->modulo;
                this->oob += err;
                errors.push_back(err);
            }

            if ( count_for_oob > 0 )
                this->oob /= count_for_oob;

            this->oob_min = errors[0];
            this->oob_max = errors[0];
            for ( size_t i = 0; i < errors.size(); i++ ) {
                if ( errors[i] < this->oob_min )
                    this->oob_min = errors[i];
                if ( errors[i] > this->oob_max )
                    this->oob_max = errors[i];
            }
            
            if ( verbose ) {
                std::cout << "Out-of-bag (OOB) error estimate:" << std::endl;
                std::cout << "Number os samples used in OOB: " << count_for_oob << std::endl;
                std::cout << "OOB (Mean error): " << this->oob << std::endl;
                std::cout << "Min. error: " << this->oob_min << std::endl;
                std::cout << "Max. error: " << this->oob_max << std::endl;
            }
        }
    }
    void grow( std::vector<PointType>& points,
               std::vector< double >& _responses,
               bool verbose ) {

        std::vector< std::vector<double> > responses(_responses.size());
        if ( this->params->modulo == -1 ) {
            for ( size_t i = 0; i < responses.size(); i++ )
                responses[i].push_back( _responses[i] );
        }
        else {
            assert( this->params->modulo > 1 );
            for ( size_t i = 0; i < responses.size(); i++ ) {
                // See:
                // [C.M. Bishop] Pattern Recognition and Machine Learning
                // (p. 105)
                // for justification
                double theta = _responses[i] / this->params->modulo * 2*CV_PI;
                responses[i].push_back( cos(theta) );
                responses[i].push_back( sin(theta) );
            }
        }

        this->grow( points, responses, verbose );
        
    }
    
    void _predict( PointType& point, std::vector<double>& predictions ) {
        // Setting maximum number of threads
        /* tbb::task_scheduler_init init( this->params->parallel ); */
        /* tbb::parallel_for(tbb::blocked_range<size_t>(0, this->size() ), */
        /*                   PredictTrees<PointType>( this->trees, */
        /*                                            point, */
        /*                                            predictions ) */
        /*                   ); */

        for ( int i = 0; i < this->size(); i++ ) {
            std::vector<double> p = this->trees[i].predict(point);
            for ( int c = 0; c < this->nb_labels; c++ )
                predictions[c] += p[c];
        }
    }


    std::vector<double> predict_regression( PointType& point ) {
        /* tbb::parallel_for(tbb::blocked_range<size_t>(0, this->ntrees ), */
        /*                   PredictTrees<PointType>( this->trees, */
        /*                                            point, */
        /*                                            predictions ) */
        /*                   ); */
        std::vector< std::vector<double> > predictions;
        for ( int i = 0; i < this->size(); i++ )
            predictions.push_back( this->trees[i].predict(point) );

        int dim = predictions[0].size();
        std::vector<double> prediction( dim, 0 );
        for ( int i = 0; i < this->size(); i++ )
            for ( int d = 0; d < dim; d++ )
                prediction[d] += predictions[i][d];

       for ( int d = 0; d < dim; d++ )
           prediction[d] /= this->size();

       if ( this->params->modulo == -1 )
           return prediction;
       else {
           assert( prediction.size() == 2 );
           /*
             http://www.cplusplus.com/reference/clibrary/cmath/acos/
             Return Value of acos
             Principal arc cosine of x, in the interval [0,pi] radians.
           */
           double norm = sqrt( prediction[0]*prediction[0] +
                               prediction[1]*prediction[1] );
           if ( norm == 0 ) {
               // Random guess
               cv::RNG rng;
               std::cout << "ZERO NORM\n";
               exit(1);
               return std::vector<double>( 1, rng.uniform( 0.0, this->params->modulo ) );
           }
           double theta = acos( prediction[0] / norm );
           theta *= sign( prediction[1] );
           if ( theta < 0 )
               theta += 2*CV_PI;
           theta = theta / (2*CV_PI) * this->params->modulo;
           return std::vector<double>( 1, theta );
       }
    }

    /* int _predict_binary_exit_early( PointType& point, int threshold ) { */
    /*     int count0 = 0; */
    /*     int count1 = 0; */
    /*     int i = 0; */
    /*     while ( ( count0 < threshold ) */
    /*             && (i < this->ntrees) ) { */
    /*         if ( this->trees[i]->predict(point) == 1 ) */
    /*             count1 += 1; */
    /*         else */
    /*             count0 += 1; */
    /*         i += 1; */
    /*     } */
    /*     if ( count0 >= threshold ) */
    /*         count1 = 0; */
    /*     return count1; */
    /* } */

    /* void _predict_exit_early( PointType& point, */
    /*                           std::vector<int>& predictions, */
    /*                           int threshold ) { */
    /*     int i = 0; */
    /*     while ( ( predictions[0] < threshold ) */
    /*             && (i < this->ntrees) ) { */
    /*         predictions[this->trees[i]->predict(point)] += 1; */
    /*         i += 1; */
    /*     } */
    /*     if ( predictions[0] >= threshold ) */
    /*         for ( int c = 1; c < predictions.size(); c++ ) */
    /*             predictions[c] = 0; */
    /* }      */
    
    int predict_hard( PointType& point ) {

        std::vector<double> predictions(this->nb_labels, 0);
        this->_predict( point, predictions );
            
        int response = -1;
        double max_count = -1;

        for ( int c = 0; c < this->nb_labels; c++ )
            if ( predictions[c] > max_count ) {
                response = c;
                max_count = predictions[c];
            }

        return response;
    }

    std::vector<double> predict_soft( PointType& point ) {

        /* response.resize(this->nb_labels); */
        /* for ( int c = 0; c < this->nb_labels; c++ ) */
        /*     response[c] = 0.0; */
        
        std::vector<double> predictions(this->nb_labels,0);
        this->_predict( point, predictions );

        for ( int c = 0; c < this->nb_labels; c++ )
            predictions[c] /= this->size();

        return predictions;
    }

    // soft binary early exit
    /* double predict_SBEE( PointType& point, double threshold ) { */
    /*     return (double)(this->_predict_binary_exit_early( point, threshold * this->ntrees )) */
    /*         / this->ntrees; */
    /* } */

    /* void predict_SEE( PointType& point, std::vector<double>& response, double threshold ) { */

    /*     /\* response.resize(this->nb_labels); *\/ */
    /*     /\* for ( int c = 0; c < this->nb_labels; c++ ) *\/ */
    /*     /\*     response[c] = 0.0; *\/ */
        
    /*     std::vector<int> predictions(this->nb_labels,0); */
    /*     this->_predict_exit_early( point, predictions, threshold * this->ntrees */
    /*     ); */
    /*     //std::cout << "prediction DONE\n"; */

    /*     for ( int c = 0; c < this->nb_labels; c++ ) */
    /*         response[c] = (double)(predictions[c]) / this->ntrees; */

    /* }     */

    void write( std::string folder ) {
        boost::filesystem::remove_all(folder);
        boost::filesystem::create_directories( folder );
        for ( int i = 0; i < this->size(); i++ ) {
            std::stringstream ss;
            ss << folder << "/" << i << ".data";
            this->trees[i].write( ss.str() );
        }
    }

    Forest( std::string folder,
            Test<PointType>* test_class,
            int parallel=-1 ) {
        
        std::vector<std::string> tree_files = glob( folder + "/*" );

        this->params = new TreeParams< PointType >( -1,
                                                    -1,
                                                    -1,
                                                    test_class );
        this->params->set_parallel(parallel);
        
        this->params->ntrees = tree_files.size();
        this->trees.resize( this->params->ntrees, Tree< PointType >( NULL, 0 ) );
        
        /* for ( int i = 0; i < this->params->ntrees; i++ ) { */
        /*     std::cout << "Reading " << tree_files[i] << "\n"; */
        /*     this->trees[i] = Tree< PointType >( tree_files[i], test_class ); */
        /* } */

        tbb::task_scheduler_init init( this->params->parallel );
        tbb::parallel_for(tbb::blocked_range<size_t>(0, this->params->ntrees ),
                          ParallelRead<PointType>( this->trees,
                                                   tree_files,
                                                   test_class )
                          );
        
        this->nb_labels = this->trees[0].get_nb_labels();
    }

    std::vector<double> get_feature_importance() {
        std::vector<double> feature_importance(this->params->nfeatures,0); 
        for ( int i = 0; i < this->size(); i++ )
            for ( int n = 0; n < this->params->nfeatures; n++ )
                feature_importance[n] += this->trees[i].feature_importance[n];

        for ( int n = 0; n < this->params->nfeatures; n++ )
            feature_importance[n] /= this->size();

        return feature_importance;
    }
};


#endif // __FOREST_H__ 
