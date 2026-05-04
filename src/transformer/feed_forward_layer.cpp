#include "feed_forward_layer.h"

#ifndef FEED_FORWARD_CPP
#define FEED_FORWARD_CPP

FeedForward::FeedForward(size_t d_inp, size_t d_out, size_t d_seq, bool activate ):
        net    (d_seq, d_out),
        d_net  (d_seq, d_out),
        act    (d_seq, d_out),
        err    (d_seq, d_out),
        d_error(d_seq, d_inp),
        bias   (1    , d_out),
        d_bias (1    , d_out), 
        w      (d_inp, d_out),
        d_w    (d_inp, d_out),
        adam   (d_inp, d_out),
        adam_b (1    , d_out),
        input  (d_seq,d_inp ),
        activate(activate) {
            
    this -> w.he_init();
    this -> bias.zero_init();
    this -> d_w.zero_init();
};

Matrix&  FeedForward::forward(Matrix& input) {
    Matrix::gemm(input, this -> w, this -> net);
    input.copy(this -> input);

    this -> net += this-> bias;
    if(!this -> activate) {
        this -> net.copy(this -> act);
    } else {
        this -> net.activate_relu(this -> act );
    }
    return this -> act;
};

Matrix&  FeedForward::backward(Matrix& gradient) {
    
    if( this -> activate )
        this -> net.leaky_relu_backward(this -> d_net);
    else 
        this -> net.copy(this -> d_net);

    Matrix::ewmm(this -> d_net, gradient, this -> err);

    this -> input.transpose();
    Matrix::gemm(this -> input,  this -> err, this -> d_w);
    this -> input.transpose();

    this -> w.transpose();
    Matrix::gemm(this -> err, this ->w, this -> d_error);
    this -> w.transpose();

    this -> err.col_sums(this -> d_bias);
    this -> adam.step(this ->d_w);   
    this -> adam_b.step(this -> d_bias);
    return this -> d_error;
};

void FeedForward::learn() {
    this -> adam.learn(this -> w );
    this -> adam_b.learn(this -> bias);
};
#endif