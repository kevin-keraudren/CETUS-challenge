#ifndef __SLIDINGWINDOW_H__
#define __SLIDINGWINDOW_H__

#include <opencv2/core/core.hpp>        // cv::Mat etc, always need this

struct SlidingWindow {

    cv::Mat* img;
    cv::Mat* gradX;
    cv::Mat* gradY;
    int x,y;
    int w,h;
    std::vector<double> histogram;
    std::vector< std::vector<double> > intensity_radial_histogram;
    std::vector< std::vector<double> > gradient_radial_histogram;
    std::vector< std::vector<double> > symmetry_radial_histogram;
    
    SlidingWindow( cv::Mat* _img ) {
        this->img = _img;
        this->x = 0;
        this->y = 0;
        this->w = this->img->cols;
        this->h = this->img->rows;
        //this->histogram.resize(0);
    }
    SlidingWindow( cv::Mat& _img ) {
        this->img = new cv::Mat(_img);
        this->x = 0;
        this->y = 0;
        this->w = this->img->cols;
        this->h = this->img->rows;
        //this->histogram.resize(0);
    }
    SlidingWindow( cv::Mat* _img, cv::Mat* _gradX, cv::Mat* _gradY ) {
        this->img = _img;
        this->gradX = _gradX;
        this->gradY = _gradY;
        this->x = 0;
        this->y = 0;
        this->w = this->img->cols;
        this->h = this->img->rows;
        //this->histogram.resize(0);
    }
    SlidingWindow( cv::Mat& _img, cv::Mat& _gradX, cv::Mat& _gradY ) {
        this->img = new cv::Mat(_img);
        this->gradX = new cv::Mat(_gradX);
        this->gradY = new cv::Mat(_gradY);
        this->x = 0;
        this->y = 0;
        this->w = this->img->cols;
        this->h = this->img->rows;
        //this->histogram.resize(0);
    }      

    void clear() {
        delete this->img;
        delete this->gradX;
        delete this->gradY;
    }
    
    //  non-static reference member ‘cv::Mat& SlidingWindow::img’, can't use
    //  default assignment operator 
/* SlidingWindow( cv::Mat &_img ) : */
/*     img(_img), x(0), y(0), w(_img.cols), h(_img.rows) */
/*     { } */
    
    inline void set ( int _x, int _y, int _w, int _h ) {
        this->x = _x;
        this->y = _y;
        this->w = _w;
        this->h = _h;
    }
    inline void set ( int _x, int _y ) {
        this->x = _x;
        this->y = _y;
    }
};

#endif // __SLIDINGWINDOW_H__
