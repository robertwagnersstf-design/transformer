#include "feed_forward_layer.h"

#ifndef FEED_FORWARD_CPP
#define FEED_FORWARD_CPP

FeedForward::FeedForward(size_t d_inp, size_t d_out, size_t d_seq, bool activate ):
        net (d_seq, d_out),
        act (d_seq, d_out),
        err (d_seq, d_out),
        bias(1    , d_out),
        w   (d_inp, d_out),
        d_w (d_inp, d_out),
        adam(d_inp, d_out),
        activate(activate) {
            
    this -> w.he_init();
    this -> bias.zero_init();
    this -> d_w.zero_init();
};

Matrix&  FeedForward::forward(Matrix& input) {
    Matrix::gemm(input, this -> w, this -> net);
    this -> net += this-> bias;
    if(!this -> activate) {
        this -> net.copy(this -> act);
    } else {
        this -> net.activate_relu(this -> act );
    }
    return this -> act;
};

#endif