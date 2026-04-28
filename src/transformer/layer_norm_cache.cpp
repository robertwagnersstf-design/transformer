#ifndef LAYER_NORM_CACHE_CPP
#define LAYER_NORM_CACHE_CPP
#include "layer_norm_cache.h"

LayerNormCache::LayerNormCache(size_t d_seq, size_t d_model ):
        means()   ,
        inv_stds(),
        normalized_input(d_seq, d_model),
        gamma  (1, d_model       ),
        beta   (1, d_model       ) {
    this -> means.resize   (d_seq, 0.);
    this -> inv_stds.resize(d_seq, 0.);
    
    std::fill(this -> gamma.data -> begin(), this -> gamma.data -> end(), 1. );
    std::fill(this -> beta.data  -> begin(), this -> beta.data  -> end(), 0. );
};

Matrix&  LayerNormCache::forward(Matrix& input) {
    input.layer_norm(this -> normalized_input, this -> beta, this -> gamma, this -> means, this -> inv_stds);
    return this -> normalized_input;
};
#endif