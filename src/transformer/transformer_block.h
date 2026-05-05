#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "multi_head_attention_layer.h"
#include "layer_norm_cache.h"
#include "feed_forward_layer.h"
#include "../consts.h"

class TransformerBlock {
public:
    TransformerBlock(size_t d_seq, size_t d_model, size_t d_heads);

    LayerNormCache ln_attention, ln_ffn;
    FeedForward expansion, final;
    
    Matrix input, ffn_input,block_output, grad_mid, grad_final;
    MultiHeadAttention mha;

    Matrix& forward(Matrix& input);
    Matrix& backward(Matrix& gradient);

    void learn();
    void step();

    void squared_gradient_sum();
    void apply_scale(float f);
};

#endif