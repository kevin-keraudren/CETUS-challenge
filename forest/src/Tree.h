#ifndef __TREE_H__
#define __TREE_H__

// Templated code needs to be in header files.
// http://stackoverflow.com/questions/8395653/c-linker-error-undefined-reference-only-to-one-specific-class-member-defined

// Reason for using a pointer for test_class and params:
// error: non-static reference member  can’t use default assignment operator

#include "Node.h"
#include "Test.h"
#include "utils.h"

#include <cmath>
#include <stdio.h>

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <cassert>
#include <cstdlib>

#include <opencv2/core/core.hpp>        // cv::Mat etc, always need this
#include <opencv2/imgproc/imgproc.hpp>  // all the image processing functions
#include <opencv2/highgui/highgui.hpp>  // Display and file I/O

std::vector<double> compute_mean( std::vector< std::vector<double> >& v ) {
    int dim = v[0].size();
    int N = v.size();
    std::vector<double> mean( dim, 0);

    for ( size_t i = 0; i < N; i++ )
        for ( int d = 0; d < dim; d++ )
            mean[d] += v[i][d];
    for ( int d = 0; d < dim; d++ )
        mean[d] /= N;
    
    return mean;
}

template <class PointType>
struct TreeParams {

    // Tree parameters
    int max_depth;
    int min_sample_count;
    int test_count;
    Test<PointType>* test_class;
    double min_regression_error;
    double modulo;
    int nfeatures;

    // Forest parameters
    double bagging;
    int ntrees;
    int parallel;
    
    TreeParams( int _max_depth,
                int _min_sample_count,
                int _test_count,
                Test<PointType>* _test_class ):
    max_depth(_max_depth), min_sample_count(_min_sample_count),
        test_count(_test_count), test_class(_test_class)
    {
        // Default parameters
        this->min_regression_error = 0;
        this->bagging = 0.4;
        this->modulo = -1;
        this->ntrees = 1;
        this->parallel = 0;
        this->nfeatures = 1;
    }

    void set_min_regression_error( double _min_regression_error ) {
        this->min_regression_error = _min_regression_error;
    }

    void set_bagging( double _bagging ) { this->bagging = _bagging; }

    void set_modulo( double m ) { this->modulo = m; }

    void set_ntrees( int n ) { this->ntrees = n; }

    void set_parallel( int nb_cpu ) { this->parallel = nb_cpu; }

    void set_nb_tests( int n ) { this->test_count = n; }

    void set_test( Test<PointType>* _test_class ) { this->test_class = _test_class; }  

    void set_min_items(int n) { this->min_sample_count = n; }

    void set_max_depth(int n) { this->max_depth = n; }

    void set_nfeatures(int n) { this->nfeatures = n; } 
    
};

template <class PointType>
std::ostream& operator<<(  std::ostream& Ostr,
                           TreeParams<PointType>* params ) {
    Ostr << params->max_depth << "\t";
    Ostr << params->min_sample_count << "\t";
    Ostr << params->test_count << "\t";
    Ostr << params->test_class->str() << std::endl;
    return Ostr;
}

//  forward declaration
template <class PointType> class Forest;

template <class PointType>
class Tree {
    friend class Forest<PointType>;
    
    TreeParams<PointType>* params; // tree parameters
    int nb_labels; // number of labels: we assume labels are always consecutive
                   // integers starting from zero
    std::vector<int> nb_samples;
    Node* root; // root node
    //bool classification;
    uint64 rng_state;
    cv::Mat inverse_covariance;
    
 public:
    std::vector<double> feature_importance;
    
 Tree( TreeParams<PointType>* _params, int _rng_state ):
    params(_params)
    {
        this->rng_state = _rng_state;
        this->nb_labels = -1;
        this->root = NULL;
        // this->classification = true;
    };

    Tree( std::string filename, Test<PointType>* test_class );

    ~Tree() {};
    
    int size() {
        if ( this->root == NULL )
            return 0;
        else
            return this->root->size();
    }

    int get_nb_labels() {
        return this->nb_labels;
    }

    double entropy( std::vector<double>& distribution );
    double MSE( std::vector< std::vector<double> >& responses ); 

    void split_points_count( std::vector<PointType>& points,
                             std::vector<int>& responses,
                             std::vector<double>& test,
                             std::vector<double>& left,
                             std::vector<double>& right );
    void split_points( std::vector<PointType>& points,
                             std::vector< std::vector<double> >& responses,
                             std::vector<double>& test,
                             std::vector< std::vector<double> >& left,
                             std::vector< std::vector<double> >& right );
    int leaf_label( std::vector<int>& distribution);
    void leaf_label( std::vector<double>& distribution);

    void grow( std::vector<PointType>& points,
               std::vector<int>& responses,
               bool verbose );
    void grow( std::vector<PointType>& points,
               std::vector< std::vector<double> >& responses,
               bool verbose );
    void grow( std::vector<PointType>& points,
               std::vector<double>& responses,
               bool verbose ) {
        std::vector< std::vector<double> > new_responses( responses.size() );
        if ( this->params->modulo == -1 ) {
            for ( size_t i = 0; i < responses.size(); i++ )
                new_responses[i].push_back( responses[i] );
        }
        else {
            assert( this->params->modulo > 1 );
            for ( size_t i = 0; i < responses.size(); i++ ) {
                // See:
                // [C.M. Bishop] Pattern Recognition and Machine Learning
                // (p.105)
                // for justification
                double theta = responses[i] / this->params->modulo * 2*CV_PI;
                new_responses[i].push_back( cos(theta) );
                new_responses[i].push_back( sin(theta) );
            }
        }
        this->grow( points, new_responses, verbose );        
    }
    
    void grow_node( Node** node,
                    std::vector<PointType>& points,
                    std::vector<int>& responses,
                    int depth,
                    cv::RNG& rng,
                    bool verbose );
    void grow_node( Node** node,
                    std::vector<PointType>& points,
                    std::vector< std::vector<double> >& responses,
                    int depth,
                    cv::RNG& rng,
                    bool verbose );

    std::vector<double> predict( PointType& point );
    std::vector<double> predict_node(Node* node, PointType& point );

    int predict_hard( PointType& point );
    
    void write( std::string filename );
    
    
};

template <class PointType>
void Tree<PointType>::write( std::string filename ) {
    std::ofstream outfile(filename.c_str());

    // write params
    outfile << this->params;
    
    // write nb_labels
    outfile << this->nb_labels << std::endl;
     
    // write tree
    outfile << this->root->write();
    
    outfile.close();
}

template <class PointType>
Tree<PointType>::Tree( std::string filename,
                       Test<PointType>* test_class ) {

    std::ifstream infile;
    infile.open(filename.c_str());
    std::string line;
    std::stringstream ss;
    
    // read params
    int max_depth;
    int min_sample_count;
    int test_count;
    std::string test_class_str;
    std::getline(infile, line);
    ss << line;
    ss >> max_depth >> min_sample_count >> test_count >> test_class_str;

    // check we have been given the right Test class
    assert( test_class_str.compare( test_class->str() ) == 0 );
    
    // create Tree params
    this->params = new TreeParams< PointType >( max_depth,
                                    min_sample_count,
                                    test_count,
                                    test_class );

    //std::cout << "params created in read"<<std::endl;

    // read nb_labels
    std::getline(infile, line);
    this->nb_labels = atoi(line.c_str());

    // read root
    this->root = new Node( infile );

    infile.close();
}

// The decision tree is grown by maximising Information Gain
template <class PointType>
double Tree<PointType>::entropy( std::vector<double>& distribution ) {
    double E = 0.0;
    float count = 0.0;
        
    for ( int i = 0; i < distribution.size(); i++ )
        count += distribution[i];
        
    if (count == 0 )
        return 0.0;
        
    else {
        for ( int i = 0; i < distribution.size(); i++ ) 
            if ( distribution[i] > 0 ) {
                double proba = distribution[i]/count;
                E -= proba*log(proba);
            }
    }
    return E;
}

template <class PointType>
void Tree<PointType>::split_points_count( std::vector<PointType>& points,
                         std::vector<int>& responses,
                         std::vector<double>& test,
                         std::vector<double>& left,
                         std::vector<double>& right ) {
    /* for ( int c = 0; c < this->nb_labels; c++ ) { */
    /*     left[c] = 0; */
    /*     right[c] = 0; */
    /* } */
    for ( int i = 0; i < points.size(); i++ ) {
        // if test passes, we go right
        if ( this->params->test_class->run( points[i], test ) )
            right[responses[i]] +=1;
        else
            // else we go left
            left[responses[i]] += 1;
    }
}

template <class PointType>
int Tree<PointType>::leaf_label( std::vector<int>& distribution) {
    int response = -1;
    double max_count = -1;
    /* for ( int c = 0; c < distribution.size(); c++ ) */
    /*     if ( (double)(distribution[c])/(*this->nb_samples)[c] > max_count ) { */
    /*         response = c; */
    /*         max_count = (double)(distribution[c])/(*this->nb_samples)[c]; */
    /*     } */
    for ( int c = 0; c < distribution.size(); c++ )
        if ( distribution[c] > max_count ) {
            response = c;
            max_count = distribution[c];
        }    
    return response;
}
template <class PointType>
void Tree<PointType>::leaf_label( std::vector<double>& distribution) {
    double sum = 0;
    for ( int c = 0; c < distribution.size(); c++ )
        sum += distribution[c];
    if (sum > 0)
        for ( int c = 0; c < distribution.size(); c++ )
            distribution[c] /= sum;
}

/**************************** Classification ****************************/
template <class PointType>
void Tree<PointType>::split_points( std::vector<PointType>& points,
                                          std::vector<std::vector<double> >& responses,
                                          std::vector<double>& test,
                                          std::vector< std::vector<double> >& left,
                                          std::vector< std::vector<double> >& right ) {
    for ( size_t i = 0; i < points.size(); i++ ) {
        // if test passes, we go right
        if ( this->params->test_class->run( points[i], test ) )
            right.push_back( responses[i] );
        else
            // else we go left
            left.push_back( responses[i] );
    }
}

template <class PointType>
void Tree<PointType>::grow( std::vector<PointType>& points,
                            std::vector<int>& responses,
                            bool verbose ) {

    this->feature_importance.resize(this->params->nfeatures,0);
            
    //this->classification = true;
    
    // set number of labels
    int max_label = -1;
    for ( int i = 0; i < responses.size(); i++ )
        if ( responses[i] > max_label )
            max_label = responses[i];
    this->nb_labels = max_label + 1;

    this->nb_samples.resize(this->nb_labels);
    for ( int i = 0; i < this->nb_samples.size(); i++ )
        this->nb_samples[i] = 0;
    for ( int i = 0; i < responses.size(); i++ )
        this->nb_samples[responses[i]] += 1;

    if (verbose) {
        std::cout << "Number of points: "
                  << points.size()
                  << std::endl;
        std::cout << "Number of labels: "
                  << this->nb_labels
                  << std::endl;
    }

    // Creating root node
    this->root = new Node();

    cv::RNG rng( this->rng_state );
    
    this->grow_node( &(this->root),
                     points,
                     responses,
                     0,
                     rng,
                     verbose ); 

    /* for ( int n = 0; n < this->params->nfeatures; n++ ) */
    /*     feature_importance[n] /= points.size(); */
    double sum = 0;
    for ( int n = 0; n < this->params->nfeatures; n++ )
        sum += feature_importance[n];
    if (sum>0)
        for ( int n = 0; n < this->params->nfeatures; n++ ) 
            feature_importance[n] /=sum;
        
}

template <class PointType>
void Tree<PointType>::grow_node( Node** node,
                                 std::vector<PointType>& points,
                                 std::vector<int>& responses,
                                 int depth,
                                 cv::RNG& rng,
                                 bool verbose ) {
    
    // Compute entropy of current node
    std::vector<double> node_distribution(this->nb_labels, 0);
    /* for ( int c = 0; c < this->nb_labels; c++ ) */
    /*     node_distribution[c] = 0; */
    for ( int i = 0; i < responses.size(); i++ )
        node_distribution[responses[i]] += 1;

    double H = this->entropy( node_distribution );
    if (verbose) {
        std::cout << "Current entropy: "
                  << H
                  << std::endl;
        std::cout << "Nb points: "
                  << points.size()
                  << std::endl;
    }
        
    if ( ( depth == this->params->max_depth )
         || ( points.size() <= this->params->min_sample_count )
         || ( H == 0 ) ) {
        //*node = new Node( this->leaf_label( node_distribution ) );
        this->leaf_label( node_distribution );
        *node = new Node( node_distribution );
        return;
    }
        
    std::vector< std::vector<double> > all_tests(this->params->test_count);
    this->params->test_class->generate_all( points, all_tests, rng);

    double best_gain = 0;
    int best_i = -1;
    for ( int i = 0; i < all_tests.size(); i++ ) {
        std::vector<double> left_points(this->nb_labels,0);
        std::vector<double> right_points(this->nb_labels,0);
        split_points_count( points,
                            responses,
                            all_tests[i],
                            left_points,
                            right_points );
                
        float left_count = 0.0;
        float right_count = 0.0;
        for (int c = 0; c < this->nb_labels; c++) {
            left_count += left_points[c];
            right_count += right_points[c];
        }

        // compute information gain
        double I = H - ( left_count/points.size()*this->entropy(left_points)
                         + right_count/points.size()*this->entropy(right_points) );
            
        // maximize information gain
        if ( I > best_gain ) {
            best_gain = I;
            best_i = i;
        }
    }

    if (verbose) {
        std::cout << "Information gain: "
                  << best_gain
                  << std::endl;
    }

    if ( best_i == -1) {
        if (verbose) {
            std::cout << "no best split found: creating a leaf"
                      << std::endl;
        }
        //*node = new Node( this->leaf_label( node_distribution ) );
        this->leaf_label( node_distribution );
        *node = new Node( node_distribution );
        return;
    }

    // Set selected test
    (*node)->make_node( all_tests[best_i] );

    // Feature importance
    this->feature_importance[this->params->test_class->feature_id(all_tests[best_i])] += best_gain*(double)(points.size());
    
    /* if (verbose) { */
    /*     std::cout << "TEST: " */
    /*               << (*node)->test */
    /*               << std::endl; */
    /* } */

    // split data
    std::vector<PointType> left_points;
    std::vector<int> left_responses;
    std::vector<PointType> right_points;
    std::vector<int> right_responses;
    for ( int i = 0; i < points.size(); i++ )
        if ( this->params->test_class->run(points[i], (*node)->test ) ) {
            right_points.push_back( points[i] );
            right_responses.push_back(responses[i]);
        }
        else {
            left_points.push_back( points[i] );
            left_responses.push_back( responses[i] );
        }

this->grow_node( &((*node)->left), left_points, left_responses,  depth+1, rng, verbose );
this->grow_node( &((*node)->right), right_points, right_responses, depth+1, rng, verbose );
    
    return;
}

template <class PointType>
inline std::vector<double> Tree<PointType>::predict( PointType& point ){
    return this->predict_node(this->root, point);
}

template <class PointType>
inline int Tree<PointType>::predict_hard( PointType& point ){
    std::vector<double> prediction = this->predict_node(this->root, point);
    int response = -1;
    double max_count = -1;

    for ( int c = 0; c < prediction.size(); c++ )
        if ( prediction[c] > max_count ) {
            response = c;
            max_count = prediction[c];
        }

        return response;
}

template <class PointType>
std::vector<double> Tree<PointType>::predict_node(Node* node, PointType& point ) {
    if (node->leaf) 
        return node->value;
    else
        if ( this->params->test_class->run( point, node->test ) )
            return this->predict_node( node->right, point);
        else
            return this->predict_node( node->left, point);
}

/**************************** Regression ****************************/

/* template <class PointType> */
/* double Tree<PointType>::MSE( std::vector< std::vector<double> >& responses ) { */
/*     int dim = responses[0].size(); */
/*     int N = responses.size(); */
/*     std::vector<double> mean = compute_mean( responses ); */

/*     double error = 0; */
/*     for ( size_t i = 0; i < N; i++ ) */
/*         for ( int d = 0; d < dim; d++ ) */
/*             error += (responses[i][d] - mean[d])*(responses[i][d] - mean[d]); */

/*     error /= N; */

/*     return error; */
/* } */

/* template <class PointType> */
/* double Tree<PointType>::MSE( std::vector< std::vector<double> >& responses ) { */
/*     int dim = responses[0].size(); */
/*     int N = responses.size(); */
/*     std::vector<double> mean = compute_mean( responses ); */

/*     double error = 0; */
/*     for ( size_t i = 0; i < N; i++ ) */
/*         for ( int d = 0; d < dim; d++ ) */
/*             error += abs(responses[i][d] - mean[d]); */

/*     //error /= N; */

/*     return error; */
/* } */

// Mean-Square Error
// using the Mahalanobis distance
// as suggested in:
// [Larsen, D.R. and Speckman, P.L.]
// Multivariate regression trees for analysis of abundance data

template <class PointType>
double Tree<PointType>::MSE( std::vector< std::vector<double> >& responses ) {
    int dim = responses[0].size();
    int N = responses.size();
    std::vector<double> mean = compute_mean( responses );

    cv::Mat error = cv::Mat::zeros( 1, 1, CV_64F );
    cv::Mat tmp = cv::Mat::zeros( dim, 1, CV_64F );
    for ( size_t i = 0; i < N; i++ ) {
        for ( int d = 0; d < dim; d++ )
            tmp.at<double>(d,0) = responses[i][d] - mean[d];
        error += tmp.t() * this->inverse_covariance * tmp;
    }

    //error /= N;
    //std::cout << "error size: " << error.rows << "  " << error.cols << "\n";
    return error.at<double>(0,0);
}

template <class PointType>
void Tree<PointType>::grow( std::vector<PointType>& points,
                            std::vector< std::vector<double> >& responses,
                            bool verbose ) {

    //this->classification = false;

    int dim = responses[0].size();
    int N = responses.size();

    //this->inverse_covariance = cv::Mat::zeros( dim, dim, CV_64F);
    this->inverse_covariance = cv::Mat::eye( dim, dim, CV_64F);

    /* // Compute mean of all responses */
    /* std::vector<double> mean = compute_mean( responses ); */

    /* // Compute covariance Matrix */
    /* cv::Mat tmp = cv::Mat::zeros( dim, 1, CV_64F ); */
    /* for ( size_t n = 0; n < N; n++ ) { */
    /*     for ( int d = 0; d < dim; d++ ) */
    /*         tmp.at<double>(d,0) = responses[n][d] - mean[d]; */
    /*             this->inverse_covariance += tmp*tmp.t(); */
    /*             } */

    /* this->inverse_covariance /= N-1; */
   
    if (verbose) {
        std::cout << "Number of points: "
                  << points.size()
                  << std::endl;
        /* std::cout << "Covariance matrix: " */
        /*           << this->inverse_covariance */
        /*           << std::endl; */
        /* std::cout << "Covariance matrix size: " */
        /*           << this->inverse_covariance.rows */
        /*     << " " << this->inverse_covariance.cols */
        /*           << std::endl; */
    }

    // inverse covariance matrix
    // the covariance matrix is positive-semidefinite and symmetric
    // default: DECOMP_LU
    // this->inverse_covariance = this->inverse_covariance.inv();//cv::DECOMP_CHOLESKY);

    // Creating root node
    this->root = new Node();

    cv::RNG rng( this->rng_state );
    
    this->grow_node( &(this->root),
                     points,
                     responses,
                     0,
                     rng,
                     verbose );
}

template <class PointType>
void Tree<PointType>::grow_node( Node** node,
                                 std::vector<PointType>& points,
                                 std::vector< std::vector<double> >& responses,
                                 int depth,
                                 cv::RNG& rng,
                                 bool verbose ) {

    double error = this->MSE( responses );
    if (verbose) {
        std::cout << "Current Mean-Square Error: "
                  << error
                  << std::endl;
    }
        
    if ( ( depth == this->params->max_depth )
         || ( points.size() <= this->params->min_sample_count )
         || ( error <= this->params->min_regression_error ) ) {
        *node = new Node( compute_mean( responses ) );
        return;
    }
        
    std::vector< std::vector<double> > all_tests(this->params->test_count);
    this->params->test_class->generate_all( points, all_tests, rng );

    double best_gain = 0;
    int best_i = -1;
    for ( size_t i = 0; i < all_tests.size(); i++ ) {
        std::vector< std::vector<double> > left;
        std::vector< std::vector<double> > right;
        split_points( points,
                      responses,
                      all_tests[i],
                      left,
                      right );

        if ( left.empty() || right.empty() )
            continue;

        double N = points.size();
        double N_left = left.size();
        double N_right = right.size();
        
        double error_left = this->MSE( left );
        double error_right = this->MSE( right );

        //std::cout << "errors " << error_left << "   " << error_right << "\n";
                
        // compute gain
        double e = error - ( error_left + error_right );
        //double e = N*error - ( N_right*error_left + N_left*error_right );

        //std::vector<double> mean_left = compute_mean( left );
        //std::vector<double> mean_right = compute_mean( right );

        //double e = 0;
        //for ( int d = 0; d < mean_left.size(); d++ )
        //    d += (mean_left[d] - mean_right[d])*(mean_left[d] - mean_right[d]);
        
        // maximize information gain
        if ( e > best_gain ) {
            best_gain = e;
            best_i = i;
        }
    }

    if (verbose) {
        std::cout << "Error gain: "
                  << best_gain
                  << std::endl;
    }

    if ( best_i == -1) {
        if (verbose) {
            std::cout << "no best split found: creating a leaf"
                      << std::endl;
        }
        *node = new Node( compute_mean( responses ) );
        return;
    }

    // Set selected test
    (*node)->make_node( all_tests[best_i] );
    
    /* if (verbose) { */
    /*     std::cout << "TEST: " */
    /*               << (*node)->test */
    /*               << std::endl; */
    /* } */

    // split data
    std::vector<PointType> left_points;
    std::vector< std::vector<double> > left_responses;
    std::vector<PointType> right_points;
    std::vector< std::vector<double> > right_responses;
    for ( int i = 0; i < points.size(); i++ )
        if ( this->params->test_class->run(points[i], (*node)->test ) ) {
            right_points.push_back( points[i] );
            right_responses.push_back(responses[i]);
        }
        else {
            left_points.push_back( points[i] );
            left_responses.push_back( responses[i] );
        }

    this->grow_node( &((*node)->left), left_points, left_responses,  depth+1, rng, verbose );
    this->grow_node( &((*node)->right), right_points, right_responses, depth+1, rng, verbose );
    
    return;
}



#endif // __TREE_H__ 
