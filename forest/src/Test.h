#ifndef __TEST_H__
#define __TEST_H__

#include <opencv2/core/core.hpp>        // cv::Mat etc, always need this

// tests are stored as double for generality

template <typename PointType>
class Test {
public:
    virtual std::string str() { return ""; }
    virtual void generate_all( std::vector<PointType>& points,
                               std::vector< std::vector<double> >& all_tests,
                               cv::RNG& rng ) { }
    virtual bool run( PointType& point, std::vector<double>& test ) {
        return false;
    }
    virtual int feature_id( std::vector<double>& test ) { return 0; }
};

#endif // __TEST_H__ 
