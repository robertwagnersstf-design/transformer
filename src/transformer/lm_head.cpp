#ifndef LM_HEAD_CPP
#define LM_HEAD_CPP
#include "lm_head.h"

LMHead::LMHead(Matrix& embeddings, size_t d_seq ): 
         embeddings  (embeddings                      ),
         d_embeddings(embeddings.rows, embeddings.cols),
         logits_cache(d_seq, embeddings.rows          ),
         d_logits    (d_seq, embeddings.rows          ),
         probs_cache (d_seq, embeddings.rows          ),
         max_idx     (d_seq                           ),
         adam        (embeddings.rows, embeddings.cols) {};

void LMHead::forward(Matrix& transformer_output) {
    this -> embeddings.transpose();

    Matrix::gemm(transformer_output, this -> embeddings, this -> logits_cache);

    this -> logits_cache.ms_softmax(this -> probs_cache);

    this -> embeddings.transpose();    
};

void LMHead::backward(std::vector<size_t>&  target, Matrix& transformer_output, Matrix& d_transformer_output) {
    this -> probs_cache.copy(this -> d_logits);

    for(size_t i = 0; i < target.size(); i++ ) {
        this -> d_logits(i, target[i]) -= 1;
    }
    this -> d_logits.transpose();
    Matrix::gemm(this -> d_logits, transformer_output, this -> d_embeddings );

    this -> d_logits.transpose();

    Matrix::gemm(this -> d_logits, this-> embeddings, d_transformer_output );
    this -> adam.step( this -> d_embeddings);
};

void LMHead::find_max_idx() {
    for(size_t i = 0; i < this -> d_logits.rows; i++) {
        float max_val = -10000.;
        size_t max_idx = 0;
        for(size_t j = 0; j < this -> d_logits.cols; i++) {
            if( this -> d_logits(i,j) > max_val) {
                max_val = d_logits(i,j);
                max_idx = j;
            }
        }
        this -> max_idx[i] = max_idx;
    }
};

#endif