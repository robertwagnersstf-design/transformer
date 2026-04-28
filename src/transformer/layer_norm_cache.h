#ifndef LAYER_NORM_CACHE_H
#define LAYER_NORM_CACHE_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "../consts.h"

class LayerNormCache {
public:
        struct AdamParams {
            Matrix m_gamma, m_beta, v_gamma, v_beta;
            int t = 0;
            AdamParams(size_t r, size_t c) : m_gamma(r, c), m_beta(r, c),  v_gamma(r, c), v_beta(r, c) {
                m_gamma.zero_init();
                m_beta.zero_init();
                v_gamma.zero_init();
                v_beta.zero_init();
            }
        };
        Matrix normalized_input; 
        Matrix gamma,beta;
        AdamParams adam;

        std::vector<float> means;
        std::vector<float> inv_stds; 

        LayerNormCache(size_t d_seq, size_t d_model);

        Matrix& forward(Matrix& input);
        Matrix& backward(Matrix& gradient);

};

#endif