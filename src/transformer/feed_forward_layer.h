#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "../consts.h"
#include "adam.h"

class FeedForward {
public:
        Matrix w, d_w; 
        Matrix net, d_net, act, err, bias,d_bias, input, d_error;
        Adam adam, adam_b;

        bool activate;

        FeedForward(size_t d_input, size_t d_output, size_t d_seq, bool activate = true);

        Matrix& forward(Matrix& input);
        Matrix& backward(Matrix& gradient );

        void learn();
};

#endif