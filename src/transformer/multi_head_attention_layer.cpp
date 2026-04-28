#include "multi_head_attention_layer.h"
#ifndef MULTI_HEAD_ATTENTION_CPP
#define MULTI_HEAD_ATTENTION_CPP

MultiHeadAttention::MultiHeadAttention(size_t d_seq, size_t d_model, size_t heads):
                    w_q    (d_model, d_model ),
                    w_k    (d_model, d_model ),
                    w_v    (d_model, d_model ),
                    w0     (d_model, d_model ),
                    d_q    (d_model, d_model ),
                    d_k    (d_model, d_model ),
                    d_v    (d_model, d_model ),
                    d_w0   (d_model, d_model ),
                    q      (d_seq  , d_model ),
                    k      (d_seq  , d_model ),
                    v      (d_seq  , d_model ),
                    d_seq  (d_seq            ),
                    d_model(d_model          ),
                    cache  (d_seq, d_model   ),
                    heads  (heads            ),
                    adam   (d_model, d_model ) {
    this -> w_q.xavier_init();
    this -> w_k.xavier_init();
    this -> w_v.xavier_init();
    this ->  w0.xavier_init();
    this -> d_q.zero_init(); 
    this -> d_k.zero_init(); 
    this -> d_v.zero_init(); 
    this -> d_w0.zero_init(); 

    for(size_t i = 0; i < heads; i++ ) {
        this -> cache.score_heads.push_back(Matrix(d_seq, d_seq) );
    }
};

void MultiHeadAttention::forward(Matrix & input) {
    input.copy(this -> cache.input);
    
    Matrix::gemm(input, this -> w_q, this -> q);
    Matrix::gemm(input, this -> w_k, this -> k);
    Matrix::gemm(input, this -> w_v, this -> v);
    
    this -> k.copy(this->cache.k);
    this -> v.copy(this->cache.v);
    this -> q.copy(this->cache.q);

    this -> k.transpose();
    Matrix::gemm(this -> q, this -> k, this -> cache.score);
    this -> k.transpose();
    
    float scale = 1.0f / std::sqrt(static_cast<float>(this -> d_model) );
    this -> cache.score *= scale;
    this -> cache.score.ms_softmax();

    Matrix::gemm(this -> cache.score, this -> v, this -> cache.output );
};

void MultiHeadAttention::forward_mha(Matrix & input) {
    input.copy(this -> cache.input);
    
    Matrix::gemm(input, this -> w_q, this -> q);
    Matrix::gemm(input, this -> w_k, this -> k);
    Matrix::gemm(input, this -> w_v, this -> v);

    this -> k.copy(this->cache.k);
    this -> v.copy(this->cache.v);
    this -> q.copy(this->cache.q);

    size_t head = 0;
    size_t head_skip = this -> d_model/this->heads;

    if(float(d_model)/float(this->heads) != head_skip) {
        throw std::runtime_error("Head not a multiple of embedding");
    }
    for(auto& score: this -> cache.score_heads) {
        Matrix s_q   = this -> q.slice(0, head * head_skip, this -> d_seq, head_skip);
        Matrix s_k   = this -> k.slice(0, head * head_skip, this -> d_seq, head_skip);
        Matrix s_v   = this -> v.slice(0, head * head_skip, this -> d_seq, head_skip);
        Matrix s_out = this -> cache.output.slice(0, head * head_skip, this -> d_seq, head_skip);

        apply_attention(s_q, s_k, s_v, score, s_out );
        ++head;
    }
    Matrix::gemm(this -> cache.output, this -> w0, this -> cache.output_w0);
};

void MultiHeadAttention::apply_attention(Matrix& s_q, Matrix& s_k, Matrix& s_v, Matrix& s_score, Matrix& s_out ) {
    s_k.transpose();
    Matrix::gemm(s_q, s_k, s_score);
    s_k.transpose();
    float scale = 1.0f / std::sqrt(static_cast<float>(this -> d_model/this->heads) );
    s_score *= scale;
    s_score.ms_softmax();
    Matrix::gemm(s_score, s_v, s_out);
};

#endif