#ifndef LAYER_NORM_CACHE_CPP
#define LAYER_NORM_CACHE_CPP
#include "layer_norm_cache.h"

LayerNormCache::LayerNormCache(size_t d_seq, size_t d_model ):
        means()   ,
        inv_stds(),
        normalized_input  (d_seq, d_model),
        d_normalized_input(d_seq, d_model),
        gamma             (1, d_model    ),
        beta              (1, d_model    ),
        d_gamma           (1, d_model    ),
        d_beta            (1, d_model    ),
        adam_gamma        (1, d_model    ),
        adam_beta         (1, d_model    )  {
    this -> means.resize   (d_seq, 0.);
    this -> inv_stds.resize(d_seq, 0.);
    this -> gamma  .value_init(1.0);
    this -> beta   .zero_init();
    this -> d_gamma.zero_init();
    this -> d_beta .zero_init();
};

Matrix&  LayerNormCache::forward(Matrix& input) {
    input.layer_norm(this -> normalized_input, this -> beta, this -> gamma, this -> means, this -> inv_stds);
    return this -> normalized_input;
};

Matrix&  LayerNormCache::backward(Matrix& gradient) {
    this -> normalized_input.layer_norm_backward(gradient, this -> d_normalized_input, this -> means, this -> inv_stds, this -> gamma );
    for(size_t j = 0; j < gradient.cols;  j++ ) {
        for(size_t i = 0; i < gradient.rows;  i++ ) {
            this -> d_beta (0,j) += gradient(i,j);
            this -> d_gamma(0,j) += gradient(i,j) * this -> normalized_input(i,j); 
        }
    }
    this -> adam_beta.step(this -> d_beta);
    this -> adam_gamma.step(this -> d_gamma);
};
#endif