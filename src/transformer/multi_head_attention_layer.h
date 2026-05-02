#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "../consts.h"
#include "adam.h"

class MultiHeadAttention {
public:
    struct AttentionCache {
        Matrix input;
        Matrix q, k, v;
        Matrix score  ;
        Matrix output ;
        Matrix d_output;
        Matrix output_w0;
        std::vector<Matrix> score_heads;
        std::vector<Matrix> d_out_heads;
        AttentionCache(size_t rows, size_t cols) 
        : input(rows, cols), q(rows, cols), k(rows, cols), v(rows, cols), score(rows, rows), output(rows, cols), d_output(rows, rows), score_heads(), d_out_heads(), output_w0(rows, cols) {}
    };

    Matrix w_q,w_k,w_v, w0 , 
           q, k, v         ,
           d_q,d_k,d_v,d_w0,
           b_q,b_k,b_v,b_w0;
    
    Adam adam_q , adam_v , adam_k , adam_w0,
         adam_bq, adam_bv, adam_bk, adam_bw0;

    size_t d_seq, d_model, heads;
    AttentionCache cache;

    MultiHeadAttention(size_t d_seq, size_t d_model, size_t heads);

    void forward(Matrix & input);
    void forward_mha(Matrix & input);
    void apply_attention(Matrix& s_q, Matrix& s_k, Matrix& s_v, Matrix& s_score, Matrix& s_out );
    void apply_backward_attention(Matrix& s_q, Matrix& s_k, Matrix& s_v, Matrix& sd_q, Matrix& sd_k, Matrix& sd_v, Matrix& soft_score, Matrix& d_score, Matrix& s_grad);
    void backward_mha(Matrix & gradient);
};

#endif