#ifndef __UTILS_H__
#define __UTILS_H__

#include <vector>
#include <iostream>
#include <glob.h>
#include <string>

#include <boost/regex.hpp>

template <class T>
std::ostream& operator<<( std::ostream& Ostr,
                          std::vector<T>& v ) {
    for ( int i = 0; i < v.size(); i++ ) { 
        Ostr << v[i];
        if ( i < v.size() - 1 )
            Ostr << "\t";
    }
    return Ostr;
}


template <class T>
std::ostream& operator<<( std::ostream& Ostr,
                          std::vector<T>* v ) {
    for ( int i = 0; i < v->size(); i++ ) { 
        Ostr << v[i];
        if ( i < v->size() - 1 )
            Ostr << "\t";
    }
    return Ostr;
}

// http://stackoverflow.com/questions/8401777/simple-glob-in-c-on-unix-system
inline std::vector<std::string> glob(const std::string& pat){
    using namespace std;
    glob_t glob_result;
    glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> ret;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        ret.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return ret;
}


struct TrainingName {
    std::string name;
    std::string raw_file;
    int r;
    int x;
    int y;
    int w;
    int h;
    int g; // flipped or not
};

std::ostream& operator<<( std::ostream& Ostr,
                          TrainingName& n ) {
    Ostr << n.name << ":\n"
         << "raw_file = " << n.raw_file << "\n"
         << "r = " << n.r << "\n"
         << "x = " << n.x << "\n"
         << "y = " << n.y << "\n"
         << "w = " << n.w << "\n"
         << "h = " << n.h << "\n"
         << "g = " << n.g << "\n\n";
        
    return Ostr;
}

void parseTrainingName( std::string name, TrainingName& result ) {
    boost::regex expression("([^/]+)_(\\d+)_(\\d+)_(\\d+)_(\\d+)_(\\d+)_([01])\\.[^.]+$");
    boost::smatch what;
    if(regex_search(name, what, expression)) {
        result.name = name;
        result.raw_file.assign( what[1].first, what[1].second );
        result.r = std::atoi(std::string(what[2]).c_str());
        result.x = std::atoi(std::string(what[3]).c_str());
        result.y = std::atoi(std::string(what[4]).c_str());
        result.w = std::atoi(std::string(what[5]).c_str());
        result.h = std::atoi(std::string(what[6]).c_str());
        result.g = std::atoi(std::string(what[7]).c_str());
        return;
    }
    std::cout << "Could not parse: " << name
              << std::endl;
    exit(1);
}

int sign( double x ) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

#endif // __UTILS_H__ 
