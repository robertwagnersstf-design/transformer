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
         d_transformer_output(d_seq, embeddings.cols  ),
         adam        (embeddings.rows, embeddings.cols) {};

Matrix& LMHead::forward(Matrix& transformer_output) {
    this -> embeddings.transpose();

    Matrix::gemm(transformer_output, this -> embeddings, this -> logits_cache);

    this -> logits_cache.ms_softmax(this -> probs_cache);

    this -> embeddings.transpose();

    return this -> probs_cache; 
};

Matrix& LMHead::backward(std::vector<size_t>&  target, Matrix& transformer_output) {
    this -> probs_cache.copy(this -> d_logits);

    d_logits.set_row(d_logits.rows - 1, 0.f);

    for(size_t i = 0; i < target.size() - 1; i++ ) {
        this -> d_logits(i, target[i + 1]) -= 1.f;
    }
    this -> d_logits.transpose();
    Matrix::gemm(this -> d_logits, transformer_output, this -> d_embeddings );

    this -> d_logits.transpose();

    Matrix::gemm(this -> d_logits, this-> embeddings, this -> d_transformer_output );
    this -> adam.step( this -> d_embeddings);

    return this -> d_transformer_output;
};

void LMHead::learn() {
    this -> adam.learn(this ->embeddings );
};

#endif