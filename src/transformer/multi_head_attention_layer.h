#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "../consts.h"

class MultiHeadAttention {
public:
    struct AttentionCache {
        Matrix input;
        Matrix q, k, v;
        Matrix score  ;
        Matrix output ;
        Matrix output_w0;
        std::vector<Matrix> score_heads;
        AttentionCache(size_t rows, size_t cols) 
        : input(rows, cols), q(rows, cols), k(rows, cols), v(rows, cols), score(rows, rows), output(rows, cols),score_heads(), output_w0(rows, cols) {}
    };

    struct AdamParams {
        Matrix m_q,m_k,m_v,m_w0,
               v_q,v_k,v_v,v_w0;
        int t = 0;
        AdamParams(size_t r, size_t c) : m_q(r, c), m_k(r, c), m_v(r,c), m_w0(r,c),v_q(r, c), v_k(r, c), v_v(r,c), v_w0(r,c) {
            m_q.zero_init();
            m_k.zero_init();
            m_v.zero_init();
            m_w0.zero_init();
            v_q.zero_init();
            v_k.zero_init();
            v_v.zero_init();
            v_w0.zero_init();
        }
    };

    Matrix w_q,w_k,w_v, w0 , 
           q, k, v         ,
           d_q,d_k,d_v,d_w0;
    
    AdamParams adam;

    size_t d_seq, d_model, heads;
    AttentionCache cache;

    MultiHeadAttention(size_t d_seq, size_t d_model, size_t heads);

    void forward(Matrix & input);
    void forward_mha(Matrix & input);
    void apply_attention(Matrix& s_q, Matrix& s_k, Matrix& s_v, Matrix& s_score, Matrix& s_out );
};

#endif