#ifndef LAYER_NORM_CACHE_H
#define LAYER_NORM_CACHE_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "../consts.h"
#include "adam.h"

class LayerNormCache {
public:
        Matrix normalized_input; 
        Matrix d_normalized_input; 
        Matrix gamma,beta;
        Matrix d_gamma,d_beta;
        Adam adam_gamma, adam_beta;

        std::vector<float> means;
        std::vector<float> inv_stds; 

        LayerNormCache(size_t d_seq, size_t d_model);

        Matrix& forward(Matrix& input);
        Matrix& backward(Matrix& gradient);

        void learn();

};

#endif