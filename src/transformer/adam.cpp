#include "adam.h"

#ifndef ADAM_CPP
#define ADAM_CPP


Adam::Adam(size_t r, size_t c):
    m  (r,c),
    v  (r,c),
    d_w(r,c)  {
        this -> m.zero_init();
        this -> v.zero_init();
        this -> d_w.zero_init();
        this -> t = 1;
    };

void Adam::store(Matrix & d_w) {
    this -> d_w += d_w;
};

void Adam::step() {
    this -> step(this -> d_w);
    this -> d_w.zero_init();
}

void Adam::step(Matrix& d_w) {
    for(size_t i = 0; i < this -> m.rows; i++ ) {
        for(size_t j = 0; j < this -> m.cols; j++ ) {
            this -> m(i,j) = BETA1*this -> m(i,j) + (1-BETA1) * d_w(i,j);
            this -> v(i,j) = BETA2*this -> v(i,j) + (1-BETA2) * d_w(i,j)*d_w(i,j);
        }
    }
}
void Adam::learn(Matrix& w) {
    
    float bc1 = 1.0f / (1.0f - pow(BETA1, this -> t));
    float bc2 = 1.0f / (1.0f - pow(BETA2, this -> t));
    
    for(size_t i = 0; i < this -> m.rows; i++ ) {
        for(size_t j = 0; j < this -> m.cols; j++ ) {
            float m = this -> m(i, j) * bc1;
            float v = this -> v(i, j) * bc2;
            w(i,j) -= m/(sqrt(v)+EPSILON) * ALPHA;
        }
    }
    this -> t++;
}
#endif