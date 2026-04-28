#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "../consts.h"

class FeedForward {
public:
        struct AdamParams {
            Matrix m_w, v_w;
            int t = 0;
            AdamParams(size_t r, size_t c) : m_w(r, c), v_w(r, c) {
                m_w.zero_init();
                v_w.zero_init();
            }
        };
        Matrix w, d_w; 
        Matrix net, act, err, bias;
        AdamParams adam;

        bool activate;

        FeedForward(size_t d_input, size_t d_output, size_t d_seq, bool activate = true);

        Matrix& forward(Matrix& input);
        Matrix& backward(Matrix& gradient);
};

#endif