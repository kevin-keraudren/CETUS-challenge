#ifndef __OPENCV_FUNCTIONS_H__
#define __OPENCV_FUNCTIONS_H__


#include <opencv2/core/core.hpp>        // cv::Mat etc, always need this
#include <opencv2/imgproc/imgproc.hpp>  // all the image processing functions
#include <opencv2/highgui/highgui.hpp>  // Display and file I/O

// http://stackoverflow.com/questions/2289690/opencv-how-to-rotate-iplimage
cv::Mat rotate( const cv::Mat& source, double angle) {
    cv::Point2f src_center(source.cols/2.0F, source.rows/2.0F);
    cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(source, dst, rot_mat, source.size());//,cv::INTER_NEAREST);
    return dst;
}

void draw_match( int x, int y, int w, int angle,
                 cv::Scalar color, cv::Mat& img ) {
    
    cv::RotatedRect rRect = cv::RotatedRect( cv::Point2f(x,y),
                                             cv::Size2f(w,w), angle );
    
    cv::Point2f vertices[4];
    rRect.points(vertices);
    for (int i = 0; i < 4; i++)
        cv::line(img, vertices[i], vertices[(i+1)%4], color);

    double rangle = (double)angle * CV_PI / 180;
    cv::line(img, cv::Point2f(x,y), cv::Point2f(x+w/2*cos(rangle),y+w/2*sin(rangle)), color);
    return;
}

#endif // __OPENCV_FUNCTIONS_H__ 
