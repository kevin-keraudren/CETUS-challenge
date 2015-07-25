#ifndef __NODE_H__
#define __NODE_H__

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#include <typeinfo>
#include "utils.h"

// /home/kevin/Training
// References or pointers?
// http://www.velocityreviews.com/forums/t449819-error-uninitialized-reference-member.html

struct Node {
    bool leaf; // is leaf or not
    //int label; // prediction label for a leaf
    Node* left; // left child
    Node* right; // right child
    std::vector<double> test; // test
    std::vector<double> value;

    /* Node(int c=-1) { */
    /*     this->leaf = true; */
    /*     this->label = c; */
    /*     this->left = NULL; */
    /*     this->right = NULL; */
    /* } */

    Node() {
        this->leaf = true;
        this->left = NULL;
        this->right = NULL;
    }    

    Node( std::vector<double> v) {
        this->leaf = true;
        //this->label = -1;
        this->value = v;
        this->left = NULL;
        this->right = NULL;
    }
    
    Node( std::ifstream& infile );

    ~Node() {}
             
    /* void make_leaf( int c ) { */
    /*     /\* std::cout << "Setting LEAF label to: " << c *\/ */
    /*     /\*           << std::endl; *\/ */
    /*     this->leaf = true; */
    /*     this->label = c; */
    /*    this->left = NULL; */
    /*     this->right = NULL;  */
    /* }     */

    void make_node( std::vector<double>& _test ) {
        //std::cout << "Making node"
        //          << std::endl;        
        this->test.resize(_test.size());
        for (int i = 0; i < _test.size(); i++ )
            this->test[i] = _test[i];
        //std::cout << "COPIED" << std::endl;
        this->leaf = false;
        //this->label = -1;
        this->left = new Node();
        this->right = new Node();
    }    

    int size() {
        if ( this->leaf ) {
            return 1;
        }
        else {
            return 1 + this->left->size() + this->right->size();
        }
    }

    void write( int line_index,
                std::vector<std::string>& buffer,
                int& next_available_id );

    std::string write();

    void load( int line_index,
               std::vector<std::string>& buffer );

    /* friend std::ostream& operator<<( std::ostream& Ostr, */
    /*                                   Node* node )  { */
    /*     Ostr << node->write(); */
    /*     return Ostr; */
    /* } */
};

void Node::write( int line_index,
                  std::vector<std::string>& buffer,
                  int& next_available_id ) {
        
    if ( this->leaf ) {
        std::stringstream ss;
        ss << "-1\t-1\t";
        /* if ( this->label > -1 ) { */
        /*     // Classification */
        /*     ss << "-1\t"; */
        /*     ss << this->label; */
        /* } */
        /* else { */
        /*     // Regression */
        /*     ss << "-2\t"; */
        for ( int i = 0; i < this->value.size(); i++ ) {
            ss << this->value[i];
            if ( i < this->value.size() - 1 )
                ss << "\t";
        }
        //}
        buffer[line_index] = ss.str();
    }
    else {
        int left_index = next_available_id;
        int right_index = next_available_id + 1;
        next_available_id += 2;
        
        std::stringstream ss;
        ss << left_index << "\t" << right_index << "\t";
        for ( int i = 0; i < this->test.size(); i++ ) {
            ss << this->test[i];
            if ( i < this->test.size() - 1 )
                ss << "\t";
        }
        buffer[line_index] = ss.str();
            
        this->left->write( left_index, buffer, next_available_id );
        this->right->write( right_index, buffer, next_available_id);
    }
}

void Node::load( int line_index,
                  std::vector<std::string>& buffer ) {
    std::stringstream ss;
    ss << buffer[line_index];
    
    int left_index;
    int right_index;
    ss >> left_index >> right_index;

    if ( left_index == -1 ) {
        this->leaf = true;
        /* if ( right_index == -1 ) { */
        /*     // Classification */
        /*     ss >> this->label; */
        /* } */
        /* else { */
        /*     // Regression */
            double v;
            while (!ss.eof()) {
                ss >> v;
                this->value.push_back(v);
            }
            //}
        return;
    }
    else {
        
        this->leaf = false;
        
        double t;
        while (!ss.eof()) {
            ss >> t;
            //std::cout << "t=" << t << std::endl;
            this->test.push_back(t);
        }
        // std::cout << "NEW TEST: " << this->test << std::endl;
        
        this->left = new Node();
        this->left->load( left_index, buffer );
        this->right = new Node();
        this->right->load( right_index, buffer );
        return;
    }
}

std::string Node::write() {
    std::vector<std::string> buffer(this->size());
    int next_available_id = 1;
    this->write( 0, buffer, next_available_id );
    std::stringstream ss;
    for ( int i = 0; i < buffer.size(); i++ ) {
        ss << buffer[i];
        if ( i < buffer.size() - 1 ) {
            ss << std::endl;
        }
    }
    return( ss.str() );
}

Node::Node( std::ifstream& infile ) {
    std::vector<std::string> buffer;
    std::string line;
    while( std::getline(infile, line) )
        buffer.push_back(line);
    this->load( 0, buffer );    
}

#endif // __NODE_H__ 
