#include "multi_head_attention_layer.h"
#ifndef MULTI_HEAD_ATTENTION_CPP
#define MULTI_HEAD_ATTENTION_CPP
//see https://jalammar.github.io/illustrated-transformer/ for insights into that mechanism
MultiHeadAttention::MultiHeadAttention(size_t d_seq, size_t d_model, size_t heads):
                    //weights
                    w_q     (d_model, d_model ),
                    w_k     (d_model, d_model ),
                    w_v     (d_model, d_model ),
                    w0      (d_model, d_model ),
                    //biases
                    b_q     (1      , d_model ),
                    b_k     (1      , d_model ),
                    b_v     (1      , d_model ),
                    b_w0    (1      , d_model ),

                    d_bq    (1      , d_model ),
                    d_bk    (1      , d_model ),
                    d_bv    (1      , d_model ),
                    d_bw0   (1      , d_model ),
                    //gradients
                    d_q     (d_seq  , d_model ),
                    d_k     (d_seq  , d_model ),
                    d_v     (d_seq  , d_model ),

                    d_wq    (d_model, d_model ),
                    d_wk    (d_model, d_model ),
                    d_wv    (d_model, d_model ),
                    d_w0    (d_model, d_model ),
                    //intermediate results
                    q       (d_seq  , d_model ),
                    k       (d_seq  , d_model ),
                    v       (d_seq  , d_model ),
                    //general params
                    d_seq   (d_seq            ),
                    d_model (d_model          ),
                    cache   (d_seq, d_model   ),
                    heads   (heads            ),
                    //adam
                    adam_q  (d_model, d_model ), 
                    adam_v  (d_model, d_model ), 
                    adam_k  (d_model, d_model ), 
                    adam_w0 (d_model, d_model ),
                    adam_bq (1      , d_model ), 
                    adam_bv (1      , d_model ), 
                    adam_bk (1      , d_model ), 
                    adam_bw0(1      , d_model )
                    {
    this -> w_q.xavier_init();
    this -> w_k.xavier_init();
    this -> w_v.xavier_init();
    this ->  w0.xavier_init();
    this -> d_q.zero_init(); 
    this -> d_k.zero_init(); 
    this -> d_v.zero_init(); 
    this -> d_w0.zero_init();
    this -> d_wq.zero_init(); 
    this -> d_wk.zero_init(); 
    this -> d_wv.zero_init(); 
    this -> d_w0.zero_init();
    this -> b_q.zero_init(); 
    this -> b_k.zero_init(); 
    this -> b_v.zero_init(); 
    this -> b_w0.zero_init();
    this -> d_bq.zero_init(); 
    this -> d_bk.zero_init(); 
    this -> d_bv.zero_init(); 
    this -> d_bw0.zero_init();   

    for(size_t i = 0; i < heads; i++ ) {
        this -> cache.score_heads.push_back(Matrix(d_seq, d_seq));
        this -> cache.d_out_heads.push_back(Matrix(d_seq, d_seq));
    }
};

void MultiHeadAttention::forward(Matrix & input) {
    input.copy(this -> cache.input);
    
    Matrix::gemm(input, this -> w_q, this -> q);
    Matrix::gemm(input, this -> w_k, this -> k);
    Matrix::gemm(input, this -> w_v, this -> v);
    
    this -> q  += this -> b_q;
    this -> k  += this -> b_k;
    this -> v  += this -> b_v;

    this -> k.copy(this->cache.k);
    this -> v.copy(this->cache.v);
    this -> q.copy(this->cache.q);

    this -> k.transpose();
    Matrix::gemm(this -> q, this -> k, this -> cache.score);
    this -> k.transpose();
    
    float scale = 1.0f / std::sqrt(static_cast<float>(this -> d_model) );
    this -> cache.score *= scale;
    // --- MASKIERUNG HIER ---
    for(size_t r = 0; r <  this -> cache.score.rows; ++r) {
        for(size_t c = r + 1; c <  this -> cache.score.cols; ++c) {
             this -> cache.score(r, c) = -1e9f;
        }
    }
    this -> cache.score.ms_softmax();

    Matrix::gemm(this -> cache.score, this -> v, this -> cache.output );
    this -> cache.output += this -> b_w0;
};

Matrix& MultiHeadAttention::forward_mha(Matrix & input) {
    input.copy(this -> cache.input);
    
    Matrix::gemm(input, this -> w_q, this -> q);
    Matrix::gemm(input, this -> w_k, this -> k);
    Matrix::gemm(input, this -> w_v, this -> v);

    this -> q  += this -> b_q;
    this -> k  += this -> b_k;
    this -> v  += this -> b_v;

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
    this -> cache.output_w0 += this -> b_w0;

    return this -> cache.output_w0;
};

void MultiHeadAttention::apply_attention(Matrix& s_q, Matrix& s_k, Matrix& s_v, Matrix& s_score, Matrix& s_out ) {
    s_k.transpose();
    Matrix::gemm(s_q, s_k, s_score);
    s_k.transpose();
    float scale = 1.0f / std::sqrt(static_cast<float>(this -> d_model/this->heads) );
    s_score *= scale;

    //apply mask
    for (size_t i = 0; i < s_score.rows; ++i) {
        for (size_t j = i + 1; j < s_score.cols; ++j) {
            s_score(i, j) = -1e9f; // Ein sehr großer negativer Wert
        }
    }
    s_score.ms_softmax();
    Matrix::gemm(s_score, s_v, s_out);
};

Matrix& MultiHeadAttention::backward_mha(Matrix & gradient) {
    
    //Last step in fwd was output time w0, so output times gradient results in delta for w0
    this -> cache.output.transpose();
    Matrix::gemm(this -> cache.output, gradient, this -> d_w0);
    this -> cache.output.transpose();

    gradient.col_sums(this-> d_bw0);

    this -> w0.transpose();
    Matrix::gemm(gradient, this -> w0, this -> cache.d_output);
    this -> w0.transpose();

    size_t head = 0;
    size_t head_skip = this -> d_model/this->heads;

    if(float(d_model)/float(this->heads) != head_skip) {
        throw std::runtime_error("Head not a multiple of embedding");
    }
    for(auto& d_score: this -> cache.d_out_heads) {
        Matrix s_q   = this -> q.slice(0, head * head_skip, this -> d_seq, head_skip);
        Matrix s_k   = this -> k.slice(0, head * head_skip, this -> d_seq, head_skip);
        Matrix s_v   = this -> v.slice(0, head * head_skip, this -> d_seq, head_skip);

        Matrix ds_q   = this -> d_q.slice(0, head * head_skip, this -> d_seq, head_skip);
        Matrix ds_k   = this -> d_k.slice(0, head * head_skip, this -> d_seq, head_skip);
        Matrix ds_v   = this -> d_v.slice(0, head * head_skip, this -> d_seq, head_skip);

        Matrix s_grad = this -> cache.d_output.slice(0, head * head_skip, this -> d_seq, head_skip);
        Matrix soft_score = this -> cache.score_heads[head];

        apply_backward_attention(s_q, s_k, s_v, ds_q, ds_k, ds_v, soft_score, d_score, s_grad );
        ++head;
    }

    this -> cache.input.transpose();
    Matrix::gemm(this -> cache.input, this -> d_q, this -> d_wq);
    Matrix::gemm(this -> cache.input, this -> d_k, this -> d_wk);
    Matrix::gemm(this -> cache.input, this -> d_v, this -> d_wv);
    this -> cache.input.transpose();

    this -> cache.d_input.zero_init();

    this -> w_q.transpose();
    Matrix::gemm(this -> d_q, this -> w_q, this -> cache.d_input);
    this -> d_q.col_sums(this -> d_bq);
    this -> w_q.transpose();

    this -> w_k.transpose();
    Matrix::gemm_accum(this -> d_k, this -> w_k, this -> cache.d_input);
    this -> d_k.col_sums(this -> d_bk);
    this -> w_k.transpose();

    this -> w_v.transpose();
    Matrix::gemm_accum(this -> d_v, this -> w_v, this -> cache.d_input);
    this -> d_v.col_sums(this -> d_bv);
    this -> w_v.transpose();
  
    return this -> cache.d_input;
};

void MultiHeadAttention::apply_backward_attention(Matrix& s_q, Matrix& s_k, Matrix& s_v, Matrix& sd_q, Matrix& sd_k, Matrix& sd_v,Matrix& soft_score, Matrix& d_score, Matrix& s_grad ) {
    // first we want to know, which participation our value matrix had in the error. Therefor we use the attentionmatirx to transfer that into it
    // this is relevant, since the attentionmatrix affected, how much a specific vector from V was considered in the output, and therefor is now represented in the error
    soft_score.transpose();
    Matrix::gemm(soft_score, s_grad, sd_v);
    soft_score.transpose();

    // now we want to move our gradient further back into the chain
    // value vector here determines, what change in the attentionmatrix would affect changes in the output
    // therefor here we get the errors of our attentionmatrix
    Matrix d_scores_raw(soft_score.rows, soft_score.cols); 
    s_v.transpose();
    Matrix::gemm(s_grad, s_v, d_scores_raw);
    s_v.transpose();
    //move it back through our softmax
    soft_score.ms_softmax_backward(d_scores_raw, d_score);
    float scale = 1.0f / std::sqrt(static_cast<float>(this -> d_model/this->heads) );
    // apply the scale since f'(a*f(b)) -> a*f'(b) 
    d_score *= scale;
    //d_score.clip_gradients();

    // now just transfer it through our s_q * s_k, to get their errors
    Matrix::gemm(d_score, s_k, sd_q);
    d_score.transpose();
    Matrix::gemm(d_score, s_q, sd_k);
    d_score.transpose();
};

void MultiHeadAttention::learn() {
    this -> adam_q.learn(this -> w_q);
    this -> adam_k.learn(this -> w_k);
    this -> adam_v.learn(this -> w_v);
    this -> adam_w0.learn(this -> w0);
    this -> adam_bq.learn(this -> b_q);
    this -> adam_bk.learn(this -> b_k);
    this -> adam_bv.learn(this -> b_v);
    this -> adam_bw0.learn(this -> b_w0);
};

void MultiHeadAttention::step() {
    this -> adam_w0.step (this -> d_w0);
    this -> adam_bw0.step(this -> d_bw0);
    this -> adam_q  .step(this -> d_wq);
    this -> adam_k  .step(this -> d_wk);
    this -> adam_v  .step(this -> d_wv);
    this -> adam_bq .step(this -> d_bq);
    this -> adam_bk .step(this -> d_bk);
    this -> adam_bv .step(this -> d_bv);
};


#endif