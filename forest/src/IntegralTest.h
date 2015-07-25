#ifndef __INTEGRALTESTS_H__
#define __INTEGRALTESTS_H__

#include <cmath>
#include <algorithm> // max, min, random_shuffle
#include <limits>

#include "Test.h"

typedef float pixeltype;
typedef float knowledgetype;

struct SlidingWindow {

    pixeltype* img;
    knowledgetype* knowledge;
    std::vector<double>* metadata;
    int shape0, shape1, shape2;
    int kshape0, kshape1, kshape2;
    int nb_knowledge_layers;
    double ksampling;
    int x,y,z,kx,ky,kz;
    
    SlidingWindow( pixeltype* _img,
                   int _shape0,
                   int _shape1,
                   int _shape2,
                   int _x,
                   int _y,
                   int _z ) {
        this->img = _img;
        this->shape0 = _shape0;
        this->shape1 = _shape1;
        this->shape2 = _shape2;
        this->x = _x;
        this->y = _y;
        this->z = _z;
    }

    SlidingWindow( pixeltype* _img,                   
                   int _shape0,
                   int _shape1,
                   int _shape2,
                   knowledgetype* _knowledge,
                   int _nb_knowledge_layers,
                   int _kshape0,
                   int _kshape1,
                   int _kshape2,
                   double _ksampling,
                   int _x,
                   int _y,
                   int _z,
                   std::vector<double>* _metadata ) {
        this->img = _img;
        this->knowledge = _knowledge;
        this->shape0 = _shape0;
        this->shape1 = _shape1;
        this->shape2 = _shape2;
        this->nb_knowledge_layers = _nb_knowledge_layers;
        this->kshape0 = _kshape0;
        this->kshape1 = _kshape1;
        this->kshape2 = _kshape2;        
        this->ksampling = _ksampling;
        this->metadata = _metadata;
        
        this->x = _x;
        this->y = _y;
        this->z = _z;
        this->kx = _x / this->ksampling;
        this->ky = _y / this->ksampling;
        this->kz = _z / this->ksampling;
    }    
    
    //  non-static reference member ‘cv::Mat& SlidingWindow::img’, can't use
    //  default assignment operator 
/* SlidingWindow( cv::Mat &_img ) : */
/*     img(_img), x(0), y(0), w(_img.cols), h(_img.rows) */
/*     { } */
    
    inline void set ( int _x, int _y, int _z ) {
        this->x = _x;
        this->y = _y;
        this->z = _z;
        this->kx = _x / this->ksampling;
        this->ky = _y / this->ksampling;
        this->kz = _z / this->ksampling;
    }

    inline size_t index( int z, int y, int x ) {
        return x + this->shape2*( y + this->shape1*z );
    }

    inline size_t index( int k, int z, int y, int x ) {
        return x + this->kshape2*( y + this->kshape1*( z + this->kshape0*k ) );
    }

    inline pixeltype mean( int cx, int cy, int cz,
                           int dx, int dy, int dz ) {
        
        int d0 = std::max(0,this->z+cz-dz);
        int r0 = std::max(0,this->y+cy-dy);
        int c0 = std::max(0,this->x+cx-dx);

        d0 = std::min(d0,this->shape0-1);
        r0 = std::min(r0,this->shape1-1);
        c0 = std::min(c0,this->shape2-1);
        
        int d1 = std::min(this->z+cz+dz,this->shape0-1);
        int r1 = std::min(this->y+cy+dy,this->shape1-1);
        int c1 = std::min(this->x+cx+dx,this->shape2-1);

        d1 = std::max(0,d1);
        r1 = std::max(0,r1);
        c1 = std::max(0,c1);

        pixeltype S = this->img[index(d1, r1, c1)];
         
        if (d0-1>=0)
            S -= this->img[index(d0-1, r1, c1)];
        if (r0-1>=0)
            S -= this->img[index(d1, r0-1, c1)];
        if (c0-1>=0)
            S -= this->img[index(d1, r1, c0-1)];
        if (r0-1>=0 && c0-1>=0)
            S += this->img[index(d1, r0-1, c0-1)];
        if (d0-1>=0 && c0-1>=0)
            S += this->img[index(d0-1, r1, c0-1)];
        if (d0-1>=0 && r0-1>=0)
            S += this->img[index(d0-1, r0-1, c1)];
        if (d0-1>=0 && r0-1>=0 && c0-1>=0)
            S -= this->img[index(d0-1, r0-1, c0-1)];
        
        return S/((d1-d0+1)*(r1-r0+1)*(c1-c0+1));
    }

    inline knowledgetype mean_knowledge( int cx, int cy, int cz,
                                 int dx, int dy, int dz,
                                 int p ) {

        cx /= this->ksampling;
        cy /= this->ksampling;
        cz /= this->ksampling;
        dx /= this->ksampling;
        dy /= this->ksampling;
        dz /= this->ksampling;

        int d0 = std::max(0,this->kz+cz-dz);
        int r0 = std::max(0,this->ky+cy-dy);
        int c0 = std::max(0,this->kx+cx-dx);

        d0 = std::min(d0,this->kshape0-1);
        r0 = std::min(r0,this->kshape1-1);
        c0 = std::min(c0,this->kshape2-1);
        
        int d1 = std::min(this->kz+cz+dz,this->kshape0-1);
        int r1 = std::min(this->ky+cy+dy,this->kshape1-1);
        int c1 = std::min(this->kx+cx+dx,this->kshape2-1);

        d1 = std::max(0,d1);
        r1 = std::max(0,r1);
        c1 = std::max(0,c1);        

        knowledgetype S = this->knowledge[index(p, d1, r1, c1)];         
        if (d0-1>=0)
            S -= this->knowledge[index(p, d0-1, r1, c1)];
        if (r0-1>=0)
            S -= this->knowledge[index(p, d1, r0-1, c1)];
        if (c0-1>=0)
            S -= this->knowledge[index(p, d1, r1, c0-1)];
        if (r0-1>=0 && c0-1>=0)
            S += this->knowledge[index(p, d1, r0-1, c0-1)];
        if (d0-1>=0 && c0-1>=0)
            S += this->knowledge[index(p, d0-1, r1, c0-1)];
        if (d0-1>=0 && r0-1>=0)
            S += this->knowledge[index(p, d0-1, r0-1, c1)];
        if (d0-1>=0 && r0-1>=0 && c0-1>=0)
            S -= this->knowledge[index(p, d0-1, r0-1, c0-1)];

        S /= ((d1-d0+1)*(r1-r0+1)*(c1-c0+1));
        
        return S;
    }

    inline void get_heart_coordinates( double &z, double &r ) {
        double z0 = (*this->metadata)[0];
        double y0 = (*this->metadata)[1];
        double x0 = (*this->metadata)[2];
        double uz = (*this->metadata)[3];
        double uy = (*this->metadata)[4];
        double ux = (*this->metadata)[5];

        double vz = this->z - z0;
        double vy = this->y - y0;
        double vx = this->x - x0;
        
        z = sqrt( vx*ux + vy*uy + vz*uz );

        double vz2 = vz - z*uz;
        double vy2 = vy - z*uy;
        double vx2 = vx - z*ux;

        r = sqrt( vx2*vx2 + vy2*vy2 + vz2*vz2 );
    }

};

std::ostream& operator<<( std::ostream& Ostr,
                          SlidingWindow& point ) {
    Ostr << "nb_knowledge_layers: " << point.nb_knowledge_layers << "\n";
    return Ostr;
}

class BlockTest: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    
public:

    BlockTest() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
    }

    std::string str() { return "BlockTest"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {

        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(12);
            int t_cx1 = rng.uniform(-this->cx,this->cx);
            int t_cy1 = rng.uniform(-this->cy,this->cy);
            int t_cz1 = rng.uniform(-this->cz,this->cz);
            int t_dx1 = rng.uniform(1,this->dx);
            int t_dy1 = rng.uniform(1,this->dy);
            int t_dz1 = rng.uniform(1,this->dz);
            int t_cx2 = rng.uniform(-this->cx,this->cx);
            int t_cy2 = rng.uniform(-this->cy,this->cy);
            int t_cz2 = rng.uniform(-this->cz,this->cz);
            int t_dx2 = rng.uniform(1,this->dx);
            int t_dy2 = rng.uniform(1,this->dy);
            int t_dz2 = rng.uniform(1,this->dz);            
            all_tests[n][0] = t_cx1;
            all_tests[n][1] = t_cy1;
            all_tests[n][2] = t_cz1;
            all_tests[n][3] = t_dx1;
            all_tests[n][4] = t_dy1;
            all_tests[n][5] = t_dz1;
            all_tests[n][6] = t_cx2;
            all_tests[n][7] = t_cy2;
            all_tests[n][8] = t_cz2;
            all_tests[n][9] = t_dx2;
            all_tests[n][10] = t_dy2;
            all_tests[n][11] = t_dz2;
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        return point.mean( test[0],
                           test[1],
                           test[2],
                           test[3],
                           test[4],
                           test[5] )  > point.mean( test[6],
                                                    test[7],
                                                    test[8],
                                                    test[9],
                                                    test[10],
                                                    test[11] );
    }

};

class AutoContext: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    int nb_knowledge_layers;
    
public:

    AutoContext() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
        this->nb_knowledge_layers = 2;
    }

    std::string str() { return "AutoContext"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    void set_nb_knowledge_layers( int _nb_knowledge_layers ) {
        this->nb_knowledge_layers = _nb_knowledge_layers;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {
        int feature;
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(13);

            feature = rng.uniform(-1,this->nb_knowledge_layers);
            all_tests[n][12] = feature;

            int t_cx1 = rng.uniform(-this->cx,this->cx);
            int t_cy1 = rng.uniform(-this->cy,this->cy);
            int t_cz1 = rng.uniform(-this->cz,this->cz);
            int t_dx1 = rng.uniform(1,this->dx);
            int t_dy1 = rng.uniform(1,this->dy);
            int t_dz1 = rng.uniform(1,this->dz);
            int t_cx2 = rng.uniform(-this->cx,this->cx);
            int t_cy2 = rng.uniform(-this->cy,this->cy);
            int t_cz2 = rng.uniform(-this->cz,this->cz);
            int t_dx2 = rng.uniform(1,this->dx);
            int t_dy2 = rng.uniform(1,this->dy);
            int t_dz2 = rng.uniform(1,this->dz);            
            all_tests[n][0] = t_cx1;
            all_tests[n][1] = t_cy1;
            all_tests[n][2] = t_cz1;
            all_tests[n][3] = t_dx1;
            all_tests[n][4] = t_dy1;
            all_tests[n][5] = t_dz1;
            all_tests[n][6] = t_cx2;
            all_tests[n][7] = t_cy2;
            all_tests[n][8] = t_cz2;
            all_tests[n][9] = t_dx2;
            all_tests[n][10] = t_dy2;
            all_tests[n][11] = t_dz2;
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        if (test[12]==-1)
            return point.mean( test[0],
                               test[1],
                               test[2],
                               test[3],
                               test[4],
                               test[5] )  > point.mean( test[6],
                                                        test[7],
                                                        test[8],
                                                        test[9],
                                                        test[10],
                                                        test[11] );
        else
            return point.mean_knowledge( test[0],
                                         test[1],
                                         test[2],
                                         test[3],
                                         test[4],
                                         test[5],
                                         test[12] ) > point.mean_knowledge( test[6],
                                                                            test[7],
                                                                            test[8],
                                                                            test[9],
                                                                            test[10],
                                                                            test[11],
                                                                            test[12] );
    }

    int feature_id( std::vector<double>& test ) {
        return test[12]+1;
    }
};

class AutoContext2: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    int nb_knowledge_layers;
    
public:

    AutoContext2() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
        this->nb_knowledge_layers = 2;
    }

    std::string str() { return "AutoContext2"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    void set_nb_knowledge_layers( int _nb_knowledge_layers ) {
        this->nb_knowledge_layers = _nb_knowledge_layers;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {
        // int image_or_knowledge;
        int feature1, feature2;
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(14);

            /* image_or_knowledge = rng.uniform(0,2); */

            /* if ( image_or_knowledge == 0 ) { */
            /*     feature1 = -1; */
            /*     feature2 = -1; */
            /* } */
            /* else { */
            /*     feature1 = rng.uniform(0,this->nb_knowledge_layers); */
            /*     feature2 = rng.uniform(0,this->nb_knowledge_layers); */
            /* } */

            feature1 = rng.uniform(-1,this->nb_knowledge_layers);
            feature2 = rng.uniform(0,this->nb_knowledge_layers);            
            
            all_tests[n][12] = feature1;
            all_tests[n][13] = feature2;

            int t_cx1 = rng.uniform(-this->cx,this->cx);
            int t_cy1 = rng.uniform(-this->cy,this->cy);
            int t_cz1 = rng.uniform(-this->cz,this->cz);
            int t_dx1 = rng.uniform(1,this->dx);
            int t_dy1 = rng.uniform(1,this->dy);
            int t_dz1 = rng.uniform(1,this->dz);
            int t_cx2 = rng.uniform(-this->cx,this->cx);
            int t_cy2 = rng.uniform(-this->cy,this->cy);
            int t_cz2 = rng.uniform(-this->cz,this->cz);
            int t_dx2 = rng.uniform(1,this->dx);
            int t_dy2 = rng.uniform(1,this->dy);
            int t_dz2 = rng.uniform(1,this->dz);            
            all_tests[n][0] = t_cx1;
            all_tests[n][1] = t_cy1;
            all_tests[n][2] = t_cz1;
            all_tests[n][3] = t_dx1;
            all_tests[n][4] = t_dy1;
            all_tests[n][5] = t_dz1;
            all_tests[n][6] = t_cx2;
            all_tests[n][7] = t_cy2;
            all_tests[n][8] = t_cz2;
            all_tests[n][9] = t_dx2;
            all_tests[n][10] = t_dy2;
            all_tests[n][11] = t_dz2;
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        if (test[12]==-1)
            return point.mean( test[0],
                               test[1],
                               test[2],
                               test[3],
                               test[4],
                               test[5] )  > point.mean( test[6],
                                                        test[7],
                                                        test[8],
                                                        test[9],
                                                        test[10],
                                                        test[11] );
        else
            return point.mean_knowledge( test[0],
                                         test[1],
                                         test[2],
                                         test[3],
                                         test[4],
                                         test[5],
                                         test[12] ) > point.mean_knowledge( test[6],
                                                                            test[7],
                                                                            test[8],
                                                                            test[9],
                                                                            test[10],
                                                                            test[11],
                                                                            test[13] );
    }

    int feature_id( std::vector<double>& test ) {
        if (test[12] == -1)
            return 0;
        else
            return 1 + std::max(test[12],test[13]) + std::min(test[12],test[13])*this->nb_knowledge_layers;
    }
};

class AutoContextN: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    int nb_knowledge_layers;
    int N;
    
public:

    AutoContextN() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
        this->nb_knowledge_layers = 2;
        this->N = 4; // must be even
    }

    std::string str() { return "AutoContextN"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    void set_N( int _N ) {
        if ( (_N % 2) != 0 ) {
            std::cout << "N must be odd in autocontextN"
                      << std::endl;
            exit(1);
        }
        this->N = _N;
    }

    void set_nb_knowledge_layers( int _nb_knowledge_layers ) {
        this->nb_knowledge_layers = _nb_knowledge_layers;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {

        int feature;
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize( this->N + 7*this->N );

            feature = rng.uniform(-1,this->nb_knowledge_layers);
            all_tests[n][0] = feature;
            for ( int feat = 1; feat < this->N; feat++ ) {
                int feature = rng.uniform(0,this->nb_knowledge_layers);
                all_tests[n][feat] = feature;
            }

            for ( int feat = 0; feat < this->N; feat++ ) {
                int t_cx = rng.uniform(-this->cx,this->cx);
                int t_cy = rng.uniform(-this->cy,this->cy);
                int t_cz = rng.uniform(-this->cz,this->cz);
                int t_dx = rng.uniform(1,this->dx);
                int t_dy = rng.uniform(1,this->dy);
                int t_dz = rng.uniform(1,this->dz);
                double t_lambda = (feat%2)==0?1.0:-1.0;
                all_tests[n][this->N + 7*feat + 0] = t_cx;
                all_tests[n][this->N + 7*feat + 1] = t_cy;
                all_tests[n][this->N + 7*feat + 2] = t_cz;
                all_tests[n][this->N + 7*feat + 3] = t_dx;
                all_tests[n][this->N + 7*feat + 4] = t_dy;
                all_tests[n][this->N + 7*feat + 5] = t_dz;
                all_tests[n][this->N + 7*feat + 6] = t_lambda;
            }
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        double res = 0.0;
        if (test[0]==-1)
            for ( int feat = 0; feat < this->N; feat++ )
                res += test[this->N + 7*feat + 6]*point.mean( test[this->N + 7*feat + 0],
                                                              test[this->N + 7*feat + 1],
                                                              test[this->N + 7*feat + 2],
                                                              test[this->N + 7*feat + 3],
                                                              test[this->N + 7*feat + 4],
                                                              test[this->N + 7*feat + 5] );
        else
            for ( int feat = 0; feat < this->N; feat++ )
                res += test[this->N + 7*feat + 6]*point.mean_knowledge( test[this->N + 7*feat + 0],
                                                                        test[this->N + 7*feat + 1],
                                                                        test[this->N + 7*feat + 2],
                                                                        test[this->N + 7*feat + 3],
                                                                        test[this->N + 7*feat + 4],
                                                                        test[this->N + 7*feat + 5],
                                                                        test[feat] );
        return res > 0;
    }

    int feature_id( std::vector<double>& test ) {
        if (test[0] == -1)
            return 0;
        else
            // TODO: FIXME
            return 1 + std::max(test[0],test[1]) + std::min(test[0],test[1])*this->nb_knowledge_layers;
    }
};

class AutoContextMetadata: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    int nb_knowledge_layers;
    
public:

    AutoContextMetadata() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
        this->nb_knowledge_layers = 2;
    }

    std::string str() { return "AutoContextMetadata"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    void set_nb_knowledge_layers( int _nb_knowledge_layers ) {
        this->nb_knowledge_layers = _nb_knowledge_layers;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {
        
        int metadata_size = (*points[0].metadata).size();
        std::vector<double> min_vector(metadata_size);
        std::vector<double> max_vector(metadata_size);
        if (metadata_size > 0) {
            for ( int i = 0; i < metadata_size; i++ ) {
                min_vector[i] = (*points[0].metadata)[i];
                max_vector[i] = (*points[0].metadata)[i];
            }
            for ( int i = 1; i < points.size(); i++ ) { // number of points
                for ( int j = 0; j < metadata_size; j++ ) { // metadata dimension
                    if ( (*points[i].metadata)[j] < min_vector[j] )
                        min_vector[j] = (*points[i].metadata)[j];
                    if ( (*points[i].metadata)[j] > max_vector[j] )
                        max_vector[j] = (*points[i].metadata)[j];
                }
            }
        }
        
        int feature;
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(13);

            if (metadata_size == 0)
                feature = rng.uniform(-1,this->nb_knowledge_layers);
            else
                feature = rng.uniform(-2,this->nb_knowledge_layers);
            
            all_tests[n][12] = feature;

            if (feature == -2) {
                int feature_metadata = rng.uniform(0,metadata_size);
                double threshold = rng.uniform(min_vector[feature_metadata],
                                               max_vector[feature_metadata]);
                all_tests[n][0] = feature_metadata;
                all_tests[n][1] = threshold;
            }
            else {
                int t_cx1 = rng.uniform(-this->cx,this->cx);
                int t_cy1 = rng.uniform(-this->cy,this->cy);
                int t_cz1 = rng.uniform(-this->cz,this->cz);
                int t_dx1 = rng.uniform(1,this->dx);
                int t_dy1 = rng.uniform(1,this->dy);
                int t_dz1 = rng.uniform(1,this->dz);
                int t_cx2 = rng.uniform(-this->cx,this->cx);
                int t_cy2 = rng.uniform(-this->cy,this->cy);
                int t_cz2 = rng.uniform(-this->cz,this->cz);
                int t_dx2 = rng.uniform(1,this->dx);
                int t_dy2 = rng.uniform(1,this->dy);
                int t_dz2 = rng.uniform(1,this->dz);            
                all_tests[n][0] = t_cx1;
                all_tests[n][1] = t_cy1;
                all_tests[n][2] = t_cz1;
                all_tests[n][3] = t_dx1;
                all_tests[n][4] = t_dy1;
                all_tests[n][5] = t_dz1;
                all_tests[n][6] = t_cx2;
                all_tests[n][7] = t_cy2;
                all_tests[n][8] = t_cz2;
                all_tests[n][9] = t_dx2;
                all_tests[n][10] = t_dy2;
                all_tests[n][11] = t_dz2;
                }
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        if (test[12]==-1)
            return point.mean( test[0],
                               test[1],
                               test[2],
                               test[3],
                               test[4],
                               test[5] )  > point.mean( test[6],
                                                        test[7],
                                                        test[8],
                                                        test[9],
                                                        test[10],
                                                        test[11] );

        else if (test[12]==-2)
            return (*point.metadata)[test[0]] > test[1];

        else
            return point.mean_knowledge( test[0],
                                         test[1],
                                         test[2],
                                         test[3],
                                         test[4],
                                         test[5],
                                         test[12] ) > point.mean_knowledge( test[6],
                                                                            test[7],
                                                                            test[8],
                                                                            test[9],
                                                                            test[10],
                                                                            test[11],
                                                                            test[12] );
    }

    int feature_id( std::vector<double>& test ) {
        return test[12]+2;
    }
};

class AdaptiveAutoContext: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    int nb_knowledge_layers;
    
public:

    AdaptiveAutoContext() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
        this->nb_knowledge_layers = 2;
    }

    std::string str() { return "AdaptiveAutoContext"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    void set_nb_knowledge_layers( int _nb_knowledge_layers ) {
        this->nb_knowledge_layers = _nb_knowledge_layers;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {      
        int feature;
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(13);

            feature = rng.uniform(-1,this->nb_knowledge_layers);
            all_tests[n][12] = feature;

            int t_cx1 = rng.uniform(-this->cx,this->cx);
            int t_cy1 = rng.uniform(-this->cy,this->cy);
            int t_cz1 = rng.uniform(-this->cz,this->cz);
            int t_dx1 = rng.uniform(1,this->dx);
            int t_dy1 = rng.uniform(1,this->dy);
            int t_dz1 = rng.uniform(1,this->dz);
            int t_cx2 = rng.uniform(-this->cx,this->cx);
            int t_cy2 = rng.uniform(-this->cy,this->cy);
            int t_cz2 = rng.uniform(-this->cz,this->cz);
            int t_dx2 = rng.uniform(1,this->dx);
            int t_dy2 = rng.uniform(1,this->dy);
            int t_dz2 = rng.uniform(1,this->dz);            
            all_tests[n][0] = t_cx1;
            all_tests[n][1] = t_cy1;
            all_tests[n][2] = t_cz1;
            all_tests[n][3] = t_dx1;
            all_tests[n][4] = t_dy1;
            all_tests[n][5] = t_dz1;
            all_tests[n][6] = t_cx2;
            all_tests[n][7] = t_cy2;
            all_tests[n][8] = t_cz2;
            all_tests[n][9] = t_dx2;
            all_tests[n][10] = t_dy2;
            all_tests[n][11] = t_dz2;
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        if (test[12]==-1)
            return point.mean( (*point.metadata)[1]*test[0],
                               (*point.metadata)[1]*test[1],
                               (*point.metadata)[1]*test[2],
                               (*point.metadata)[1]*test[3],
                               (*point.metadata)[1]*test[4],
                               (*point.metadata)[1]*test[5] )  > point.mean( (*point.metadata)[1]*test[6],
                                                                             (*point.metadata)[1]*test[7],
                                                                             (*point.metadata)[1]*test[8],
                                                                             (*point.metadata)[1]*test[9],
                                                                             (*point.metadata)[1]*test[10],
                                                                             (*point.metadata)[1]*test[11] );
        else
            return point.mean_knowledge( (*point.metadata)[1]*test[0],
                                         (*point.metadata)[1]*test[1],
                                         (*point.metadata)[1]*test[2],
                                         (*point.metadata)[1]*test[3],
                                         (*point.metadata)[1]*test[4],
                                         (*point.metadata)[1]*test[5],
                                         test[12] ) > point.mean_knowledge( (*point.metadata)[1]*test[6],
                                                                            (*point.metadata)[1]*test[7],
                                                                            (*point.metadata)[1]*test[8],
                                                                            (*point.metadata)[1]*test[9],
                                                                            (*point.metadata)[1]*test[10],
                                                                            (*point.metadata)[1]*test[11],
                                                                            test[12] );
    }

    int feature_id( std::vector<double>& test ) {
        return test[12]+1;
    }
};

class PatchTest: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    
public:

    PatchTest() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
    }

    std::string str() { return "PatchTest"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {
        
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(8);
            int t_cx1 = rng.uniform(-this->cx,this->cx);
            int t_cy1 = rng.uniform(-this->cy,this->cy);
            int t_cz1 = rng.uniform(-this->cz,this->cz);
            int t_dx1 = rng.uniform(1,this->dx);
            int t_dy1 = rng.uniform(1,this->dy);
            int t_dz1 = rng.uniform(1,this->dz);
            int threshold1 = rng.uniform(0,255);
            int threshold2 = rng.uniform(threshold1,255);           
            all_tests[n][0] = t_cx1;
            all_tests[n][1] = t_cy1;
            all_tests[n][2] = t_cz1;
            all_tests[n][3] = t_dx1;
            all_tests[n][4] = t_dy1;
            all_tests[n][5] = t_dz1;
            all_tests[n][6] = threshold1;
            all_tests[n][7] = threshold2;
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        pixeltype m = point.mean( test[0],
                                  test[1],
                                  test[2],
                                  test[3],
                                  test[4],
                                  test[5] );
        return ((m >= test[6]) && (m <= test[7]));
    }

};

class Block2DAutoContext: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    int nb_knowledge_layers;
    
public:

    Block2DAutoContext() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
        this->nb_knowledge_layers = 2;
    }

    std::string str() { return "Block2DAutoContext"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    void set_nb_knowledge_layers( int _nb_knowledge_layers ) {
        this->nb_knowledge_layers = _nb_knowledge_layers;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {
        int feature;
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(13);

            feature = rng.uniform(-1,this->nb_knowledge_layers);
            all_tests[n][12] = feature;

            int t_cx1 = rng.uniform(-this->cx,this->cx);
            int t_cy1 = rng.uniform(-this->cy,this->cy);
            int t_cz1 = rng.uniform(-this->cz,this->cz);
            int t_dx1 = rng.uniform(1,this->dx);
            int t_dy1 = rng.uniform(1,this->dy);
            int t_dz1 = rng.uniform(1,this->dz);
            int t_cx2 = rng.uniform(-this->cx,this->cx);
            int t_cy2 = rng.uniform(-this->cy,this->cy);
            int t_cz2 = rng.uniform(-this->cz,this->cz);
            int t_dx2 = rng.uniform(1,this->dx);
            int t_dy2 = rng.uniform(1,this->dy);
            int t_dz2 = rng.uniform(1,this->dz);            
            all_tests[n][0] = t_cx1;
            all_tests[n][1] = t_cy1;
            all_tests[n][2] = t_cz1;
            all_tests[n][3] = t_dx1;
            all_tests[n][4] = t_dy1;
            all_tests[n][5] = t_dz1;
            all_tests[n][6] = t_cx2;
            all_tests[n][7] = t_cy2;
            all_tests[n][8] = t_cz2;
            all_tests[n][9] = t_dx2;
            all_tests[n][10] = t_dy2;
            all_tests[n][11] = t_dz2;
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        if (test[12]==-1)
            return point.mean( test[0],
                               test[1],
                               0,
                               test[3],
                               test[4],
                               1 )  > point.mean( test[6],
                                                  test[7],
                                                  0,
                                                  test[9],
                                                  test[10],
                                                  1 );
        else
            return point.mean_knowledge( test[0],
                                         test[1],
                                         test[2],
                                         test[3],
                                         test[4],
                                         test[5],
                                         test[12] ) > point.mean_knowledge( test[6],
                                                                            test[7],
                                                                            test[8],
                                                                            test[9],
                                                                            test[10],
                                                                            test[11],
                                                                            test[12] );
    }

};

class AutoContextDistancePrior: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    int nb_knowledge_layers;
    
public:

    AutoContextDistancePrior() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
        this->nb_knowledge_layers = 2;
    }

    std::string str() { return "AutoContextDistancePrior"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    void set_nb_knowledge_layers( int _nb_knowledge_layers ) {
        this->nb_knowledge_layers = _nb_knowledge_layers;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {
        int feature;
        double max_d = 0;
        double d;
        for ( int i = 1; i < points.size(); i++ ) {
            d = std::max(std::max(points[i].shape0,points[i].shape1),points[i].shape2)/2;
            d = d*d;
            if (d>max_d)
                max_d = d;
        }
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(13);

            feature = rng.uniform(-2,this->nb_knowledge_layers);
            all_tests[n][12] = feature;

            if (feature==-2) {
                double d = rng.uniform(0.0,max_d);
                all_tests[n][0] = d;
            }
            else {
                int t_cx1 = rng.uniform(-this->cx,this->cx);
                int t_cy1 = rng.uniform(-this->cy,this->cy);
                int t_cz1 = rng.uniform(-this->cz,this->cz);
                int t_dx1 = rng.uniform(1,this->dx);
                int t_dy1 = rng.uniform(1,this->dy);
                int t_dz1 = rng.uniform(1,this->dz);
                int t_cx2 = rng.uniform(-this->cx,this->cx);
                int t_cy2 = rng.uniform(-this->cy,this->cy);
                int t_cz2 = rng.uniform(-this->cz,this->cz);
                int t_dx2 = rng.uniform(1,this->dx);
                int t_dy2 = rng.uniform(1,this->dy);
                int t_dz2 = rng.uniform(1,this->dz);            
                all_tests[n][0] = t_cx1;
                all_tests[n][1] = t_cy1;
                all_tests[n][2] = t_cz1;
                all_tests[n][3] = t_dx1;
                all_tests[n][4] = t_dy1;
                all_tests[n][5] = t_dz1;
                all_tests[n][6] = t_cx2;
                all_tests[n][7] = t_cy2;
                all_tests[n][8] = t_cz2;
                all_tests[n][9] = t_dx2;
                all_tests[n][10] = t_dy2;
                all_tests[n][11] = t_dz2;
            }
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        if (test[12]==-1)
            return point.mean( test[0],
                               test[1],
                               test[2],
                               test[3],
                               test[4],
                               test[5] )  > point.mean( test[6],
                                                        test[7],
                                                        test[8],
                                                        test[9],
                                                        test[10],
                                                        test[11] );
        else if (test[12]==-2) {
            double d =
                (point.z-point.shape0/2)*(point.z-point.shape0/2) +
                (point.y-point.shape1/2)*(point.y-point.shape1/2) +
                (point.x-point.shape2/2)*(point.x-point.shape2/2);
            return d > test[0];
        }
        else
            return point.mean_knowledge( test[0],
                                         test[1],
                                         test[2],
                                         test[3],
                                         test[4],
                                         test[5],
                                         test[12] ) > point.mean_knowledge( test[6],
                                                                            test[7],
                                                                            test[8],
                                                                            test[9],
                                                                            test[10],
                                                                            test[11],
                                                                            test[12] );
    }

    int feature_id( std::vector<double>& test ) {
        return test[12]+2;
    }
};

class AutoContextGradient: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    int nb_knowledge_layers;
    
public:

    AutoContextGradient() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
        this->nb_knowledge_layers = 2;
    }

    std::string str() { return "AutoContextGradient"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    void set_nb_knowledge_layers( int _nb_knowledge_layers ) {
        this->nb_knowledge_layers = _nb_knowledge_layers;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {
        int feature;
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(13);

            feature = rng.uniform(-2,this->nb_knowledge_layers-3);
            all_tests[n][12] = feature;

            int t_cx1 = rng.uniform(-this->cx,this->cx);
            int t_cy1 = rng.uniform(-this->cy,this->cy);
            int t_cz1 = rng.uniform(-this->cz,this->cz);
            int t_dx1 = rng.uniform(1,this->dx);
            int t_dy1 = rng.uniform(1,this->dy);
            int t_dz1 = rng.uniform(1,this->dz);
            int t_cx2 = rng.uniform(-this->cx,this->cx);
            int t_cy2 = rng.uniform(-this->cy,this->cy);
            int t_cz2 = rng.uniform(-this->cz,this->cz);
            int t_dx2 = rng.uniform(1,this->dx);
            int t_dy2 = rng.uniform(1,this->dy);
            int t_dz2 = rng.uniform(1,this->dz);            
            all_tests[n][0] = t_cx1;
            all_tests[n][1] = t_cy1;
            all_tests[n][2] = t_cz1;
            all_tests[n][3] = t_dx1;
            all_tests[n][4] = t_dy1;
            all_tests[n][5] = t_dz1;
            all_tests[n][6] = t_cx2;
            all_tests[n][7] = t_cy2;
            all_tests[n][8] = t_cz2;
            all_tests[n][9] = t_dx2;
            all_tests[n][10] = t_dy2;
            all_tests[n][11] = t_dz2;
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        if (test[12]==-1)
            return point.mean( test[0],
                               test[1],
                               test[2],
                               test[3],
                               test[4],
                               test[5] )  > point.mean( test[6],
                                                        test[7],
                                                        test[8],
                                                        test[9],
                                                        test[10],
                                                        test[11] );
        else if (test[12]==-2) {
            double gx1 = point.mean_knowledge( test[0],
                                               test[1],
                                               test[2],
                                               test[3],
                                               test[4],
                                               test[5],
                                               point.nb_knowledge_layers-3);
            double gy1 = point.mean_knowledge( test[0],
                                               test[1],
                                               test[2],
                                               test[3],
                                               test[4],
                                               test[5],
                                               point.nb_knowledge_layers-2 );
            double gz1 = point.mean_knowledge( test[0],
                                               test[1],
                                               test[2],
                                               test[3],
                                               test[4],
                                               test[5],
                                               point.nb_knowledge_layers-1 );
            double gx2 = point.mean_knowledge( test[6],
                                               test[7],
                                               test[8],
                                               test[9],
                                               test[10],
                                               test[11],
                                               point.nb_knowledge_layers-3 );
            double gy2 = point.mean_knowledge( test[6],
                                               test[7],
                                               test[8],
                                               test[9],
                                               test[10],
                                               test[11],
                                               point.nb_knowledge_layers-2 );
            double gz2 = point.mean_knowledge( test[6],
                                               test[7],
                                               test[8],
                                               test[9],
                                               test[10],
                                               test[11],
                                               point.nb_knowledge_layers-1 );
            return (gx1*gx2 + gy1*gy2 + gz1*gz2) > 0;
        }
        else
            return point.mean_knowledge( test[0],
                                         test[1],
                                         test[2],
                                         test[3],
                                         test[4],
                                         test[5],
                                         test[12] ) > point.mean_knowledge( test[6],
                                                                            test[7],
                                                                            test[8],
                                                                            test[9],
                                                                            test[10],
                                                                            test[11],
                                                                            test[12] );
    }

    int feature_id( std::vector<double>& test ) {
        return test[12]+2;
    }
};

class AutoContextGradientDistancePrior: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    int nb_knowledge_layers;
    
public:

    AutoContextGradientDistancePrior() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
        this->nb_knowledge_layers = 2;
    }

    std::string str() { return "AutoContextGradientDistancePrior"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    void set_nb_knowledge_layers( int _nb_knowledge_layers ) {
        this->nb_knowledge_layers = _nb_knowledge_layers;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {
        int feature;
        double max_d = 0;
        double d;
        for ( int i = 1; i < points.size(); i++ ) {
            d = std::max(std::max(points[i].shape0,points[i].shape1),points[i].shape2)/2;
            d = d*d;
            if (d>max_d)
                max_d = d;
        }
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(13);

            feature = rng.uniform(-3,this->nb_knowledge_layers-3);
            all_tests[n][12] = feature;

            if (feature==-3) {
                double d = rng.uniform(0.0,max_d);
                all_tests[n][0] = d;
            }
            else {
                int t_cx1 = rng.uniform(-this->cx,this->cx);
                int t_cy1 = rng.uniform(-this->cy,this->cy);
                int t_cz1 = rng.uniform(-this->cz,this->cz);
                int t_dx1 = rng.uniform(1,this->dx);
                int t_dy1 = rng.uniform(1,this->dy);
                int t_dz1 = rng.uniform(1,this->dz);
                int t_cx2 = rng.uniform(-this->cx,this->cx);
                int t_cy2 = rng.uniform(-this->cy,this->cy);
                int t_cz2 = rng.uniform(-this->cz,this->cz);
                int t_dx2 = rng.uniform(1,this->dx);
                int t_dy2 = rng.uniform(1,this->dy);
                int t_dz2 = rng.uniform(1,this->dz);            
                all_tests[n][0] = t_cx1;
                all_tests[n][1] = t_cy1;
                all_tests[n][2] = t_cz1;
                all_tests[n][3] = t_dx1;
                all_tests[n][4] = t_dy1;
                all_tests[n][5] = t_dz1;
                all_tests[n][6] = t_cx2;
                all_tests[n][7] = t_cy2;
                all_tests[n][8] = t_cz2;
                all_tests[n][9] = t_dx2;
                all_tests[n][10] = t_dy2;
                all_tests[n][11] = t_dz2;
            }
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        if (test[12]==-1)
            return point.mean( test[0],
                               test[1],
                               test[2],
                               test[3],
                               test[4],
                               test[5] )  > point.mean( test[6],
                                                        test[7],
                                                        test[8],
                                                        test[9],
                                                        test[10],
                                                        test[11] );
        else if (test[12]==-2) {
            double gx1 = point.mean_knowledge( test[0],
                                               test[1],
                                               test[2],
                                               test[3],
                                               test[4],
                                               test[5],
                                               point.nb_knowledge_layers-3);
            double gy1 = point.mean_knowledge( test[0],
                                               test[1],
                                               test[2],
                                               test[3],
                                               test[4],
                                               test[5],
                                               point.nb_knowledge_layers-2 );
            double gz1 = point.mean_knowledge( test[0],
                                               test[1],
                                               test[2],
                                               test[3],
                                               test[4],
                                               test[5],
                                               point.nb_knowledge_layers-1 );
            double gx2 = point.mean_knowledge( test[6],
                                               test[7],
                                               test[8],
                                               test[9],
                                               test[10],
                                               test[11],
                                               point.nb_knowledge_layers-3 );
            double gy2 = point.mean_knowledge( test[6],
                                               test[7],
                                               test[8],
                                               test[9],
                                               test[10],
                                               test[11],
                                               point.nb_knowledge_layers-2 );
            double gz2 = point.mean_knowledge( test[6],
                                               test[7],
                                               test[8],
                                               test[9],
                                               test[10],
                                               test[11],
                                               point.nb_knowledge_layers-1 );
            return (gx1*gx2 + gy1*gy2 + gz1*gz2) > 0;
        }
        else if (test[12]==-3) {
            double d =
                (point.z-point.shape0/2)*(point.z-point.shape0/2) +
                (point.y-point.shape1/2)*(point.y-point.shape1/2) +
                (point.x-point.shape2/2)*(point.x-point.shape2/2);
            return d > test[0];
        }
        else
            return point.mean_knowledge( test[0],
                                         test[1],
                                         test[2],
                                         test[3],
                                         test[4],
                                         test[5],
                                         test[12] ) > point.mean_knowledge( test[6],
                                                                            test[7],
                                                                            test[8],
                                                                            test[9],
                                                                            test[10],
                                                                            test[11],
                                                                            test[12] );
    }

    int feature_id( std::vector<double>& test ) {
        return test[12]+3;
    }
};

class HeartAutoContext: public Test< SlidingWindow > {
    int cx,cy,cz,dx,dy,dz;
    int nb_knowledge_layers;
    
public:

    HeartAutoContext() {
        this->cx = 50;
        this->cy = 50;
        this->cz = 50;
        this->dx = 50;
        this->dy = 50;
        this->dz = 50;
        this->nb_knowledge_layers = 2;
    }

    std::string str() { return "HeartAutoContext"; }

    void set_size( int _cx, int _cy, int _cz,
                   int _dx, int _dy, int _dz ) {
        this->cx = _cx;
        this->cy = _cy;
        this->cz = _cz;
        this->dx = _dx;
        this->dy = _dy;
        this->dz = _dz;
    }

    void set_nb_knowledge_layers( int _nb_knowledge_layers ) {
        this->nb_knowledge_layers = _nb_knowledge_layers;
    }

    inline void generate_all( std::vector< SlidingWindow >& points,
                              std::vector< std::vector<double> >& all_tests,
                              cv::RNG& rng ) {

        double r,z;
        double min_z, max_z,min_r,max_r;
        min_z = std::numeric_limits<double>::max();
        max_z = std::numeric_limits<double>::min();
        min_r = std::numeric_limits<double>::max();
        max_r = std::numeric_limits<double>::min();
        
        for ( int i = 0; i < points.size(); i++ ) {
            points[i].get_heart_coordinates(z,r);
            min_z = std::min(z,min_z);
            max_z = std::max(z,max_z);
            min_r = std::min(r,min_r);
            max_r = std::max(r,max_r);
        }
        
        int feature;
        for ( int n = 0; n < all_tests.size(); n++ ) {
            all_tests[n].resize(13);

            feature = rng.uniform(-3,this->nb_knowledge_layers);
            all_tests[n][12] = feature;

            if (feature == -2) {
                double t_z = rng.uniform(min_z,max_z);
                all_tests[n][0] = t_z;
            }
            else if (feature == -3) {
                double t_r = rng.uniform(min_r,max_r);
                all_tests[n][0] = t_r;
            }
            else {
                int t_cx1 = rng.uniform(-this->cx,this->cx);
                int t_cy1 = rng.uniform(-this->cy,this->cy);
                int t_cz1 = rng.uniform(-this->cz,this->cz);
                int t_dx1 = rng.uniform(1,this->dx);
                int t_dy1 = rng.uniform(1,this->dy);
                int t_dz1 = rng.uniform(1,this->dz);
                int t_cx2 = rng.uniform(-this->cx,this->cx);
                int t_cy2 = rng.uniform(-this->cy,this->cy);
                int t_cz2 = rng.uniform(-this->cz,this->cz);
                int t_dx2 = rng.uniform(1,this->dx);
                int t_dy2 = rng.uniform(1,this->dy);
                int t_dz2 = rng.uniform(1,this->dz);            
                all_tests[n][0] = t_cx1;
                all_tests[n][1] = t_cy1;
                all_tests[n][2] = t_cz1;
                all_tests[n][3] = t_dx1;
                all_tests[n][4] = t_dy1;
                all_tests[n][5] = t_dz1;
                all_tests[n][6] = t_cx2;
                all_tests[n][7] = t_cy2;
                all_tests[n][8] = t_cz2;
                all_tests[n][9] = t_dx2;
                all_tests[n][10] = t_dy2;
                all_tests[n][11] = t_dz2;
            }
        }
        return;
    }

    inline bool run( SlidingWindow& point, std::vector<double>& test) {
        if (test[12]==-1)
            return point.mean( test[0],
                               test[1],
                               test[2],
                               test[3],
                               test[4],
                               test[5] )  > point.mean( test[6],
                                                        test[7],
                                                        test[8],
                                                        test[9],
                                                        test[10],
                                                        test[11] );
        else if (test[12]==-2) {
            double z,r;
            point.get_heart_coordinates(z,r);
            return z > test[0];
        }
        else if (test[12]==-3) {
            double z,r;
            point.get_heart_coordinates(z,r);
            return r > test[0];
        }
        else
            return point.mean_knowledge( test[0],
                                         test[1],
                                         test[2],
                                         test[3],
                                         test[4],
                                         test[5],
                                         test[12] ) > point.mean_knowledge( test[6],
                                                                            test[7],
                                                                            test[8],
                                                                            test[9],
                                                                            test[10],
                                                                            test[11],
                                                                            test[12] );
    }

    int feature_id( std::vector<double>& test ) {
        return test[12]+3;
    }
};

#endif // __INTEGRALTESTS_H__



