#ifndef __SIMPLETESTS_H__
#define __SIMPLETESTS_H__

#include <cmath>
#include <algorithm> // max, min

#include "Test.h"
#include <opencv2/core/core.hpp>        // cv::Mat etc, always need this

class AxisAligned: public Test< std::vector<float> > {
    
public:
    AxisAligned() { }
    
    std::string str() { return "AxisAligned"; }
    
    void generate_all( std::vector< std::vector<float> >& points,
                       std::vector< std::vector<double> >& all_tests,
                       cv::RNG& rng ) {
        
        std::vector<float> min_vector(points[0].size());
        std::vector<float> max_vector(points[0].size());
        for ( int i = 0; i < points[0].size(); i++ ) {
            min_vector[i] = points[0][i];
            max_vector[i] = points[0][i];
        }

        for ( int i = 1; i < points.size(); i++ ) { // number of points
            for ( int j = 0; j < points[0].size(); j++ ) { // space dimension
                if ( points[i][j] < min_vector[j] )
                    min_vector[j] = points[i][j];
                if ( points[i][j] > max_vector[j] )
                    max_vector[j] = points[i][j];
            }
        }

        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(3);
            int axis = rng.uniform(-1,points[0].size());
            all_tests[n][0] = axis;
            if (axis == -1) {
                int axisA = rng.uniform(0,points[0].size());
                int axisB = rng.uniform(0,points[0].size());
                all_tests[n][1] = axisA;
                all_tests[n][2] = axisB;
            }
            else {
                all_tests[n][1] = rng.uniform(min_vector[axis],max_vector[axis]);
                all_tests[n][2] = -1;
            }
        }
        return;
    }


    inline bool run( std::vector<float>& point, std::vector<double>& test) {
        if (test[0]==-1)
            return point[test[1]] > point[test[2]];
        else
            return abs(point[test[0]]) > abs(test[1]);
    }

};

class Linear: public Test< std::vector<float> > {
    
public:
    Linear() { }
    
    std::string str() { return "Linear"; }
    
    void generate_all( std::vector< std::vector<float> >& points,
                       std::vector< std::vector<double> >& all_tests,
                       cv::RNG& rng ) {

        std::vector<float> min_vector(points[0].size(),0);
        std::vector<float> max_vector(points[0].size(),0);
        for ( int i = 0; i < points[0].size(); i++ ) {
            min_vector[i] = points[0][i];
            max_vector[i] = points[0][i];
        }
        
        for ( int i = 1; i < points.size(); i++ ) { // number of points
            for ( int j = 0; j < points[0].size(); j++ ) { // space dimension
                if ( points[i][j] < min_vector[j] )
                    min_vector[j] = points[i][j];
                if ( points[i][j] > max_vector[j] )
                    max_vector[j] = points[i][j];
            }
        }

        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n] = std::vector<double>(3);
            all_tests[n][0] = rng.uniform(min_vector[0],max_vector[0]);
            all_tests[n][1] = rng.uniform(min_vector[1],max_vector[1]);
            all_tests[n][2] = rng.uniform(0.0,360.0);
        }
        return;
    }


    inline bool run( std::vector<float>& point, std::vector<double>& test) {
        double theta = test[2]*CV_PI/180;
        return ( cos(theta)*(point[0]-test[0])
                 + sin(theta)*(point[1]-test[1]) ) > 0;
    }

};

class LinearND: public Test< std::vector<float> > {
    
public:
    LinearND() { }
    
    std::string str() { return "LinearND"; }
    
    void generate_all( std::vector< std::vector<float> >& points,
                       std::vector< std::vector<double> >& all_tests,
                       cv::RNG& rng ) {

        int dim = points[0].size();
        std::vector<float> min_vector(dim,0);
        std::vector<float> max_vector(dim,0);
        for ( int i = 0; i < dim; i++ ) {
            min_vector[i] = points[0][i];
            max_vector[i] = points[0][i];
        }
        
        for ( int i = 1; i < points.size(); i++ ) { // number of points
            for ( int j = 0; j < dim; j++ ) { // space dimension
                if ( points[i][j] < min_vector[j] )
                    min_vector[j] = points[i][j];
                if ( points[i][j] > max_vector[j] )
                    max_vector[j] = points[i][j];
            }
        }

        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n] = std::vector<double>(dim+1);
            for ( int i = 0; i < dim; i++ )
                all_tests[n][i] = rng.uniform(min_vector[i],max_vector[i]);
            int p = rng.uniform(0,points.size());
            double res = 0.0;
            for ( int i = 0; i < dim; i++ )
                res += points[p][i]*all_tests[n][i];
            all_tests[n][dim] = res;
        }
        return;
    }


    inline bool run( std::vector<float>& point, std::vector<double>& test) {
        double res = 0.0;
        for ( int i = 0; i < point.size(); i++ )
            res += point[i]*test[i];
        return res > test[test.size()-1];
    }

};

class Parabola: public Test< std::vector<float> > {
    
public:
    Parabola() { }
    
    std::string str() { return "Parabola"; }

    void generate_all( std::vector< std::vector<float> >& points,
                       std::vector< std::vector<double> >& all_tests,
                       cv::RNG& rng ) {

        std::vector<float> min_vector(points[0].size(),0);
        std::vector<float> max_vector(points[0].size(),0);
        for ( int i = 0; i < points[0].size(); i++ ) {
            min_vector[i] = points[0][i];
            max_vector[i] = points[0][i];
        }        

        for ( int i = 1; i < points.size(); i++ ) { // number of points
            for ( int j = 0; j < points[0].size(); j++ ) { // space dimension
                if ( points[i][j] < min_vector[j] )
                    min_vector[j] = points[i][j];
                if ( points[i][j] > max_vector[j] )
                    max_vector[j] = points[i][j];
            }
        }

        double scale = std::max(max_vector[0]-min_vector[0],
                           max_vector[1]-min_vector[1]);
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n] = std::vector<double>(4);
            all_tests[n][0] = rng.uniform(2*min_vector[0],2*max_vector[0]);
            all_tests[n][1] = rng.uniform(2*min_vector[1],2*max_vector[1]);
            all_tests[n][2] = rng.uniform(-scale,scale);
            all_tests[n][3] = rng.uniform(0,1);
        }
        return;
    }


    inline bool run( std::vector<float>& point, std::vector<double>& test) {
        double x = (point[0]-test[0]);
        double y = (point[1]-test[1]);
        double p = test[2];
        int axis = test[3];
        if ( axis == 0 )
            return x*x < p*y;
        else
            return y*y < p*x;
    }

};

#endif // __SIMPLETESTS_H__



