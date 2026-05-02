#include "transformer.h"

#ifndef TRANSFORMER_CPP
#define TRANSFORMER_CPP

Transformer::Transformer(size_t d_seq                            ,  
                         size_t d_model                          , 
                         size_t d_heads                          ,  
                         size_t repetitions                      , 
                         std::vector<std::string> dictionary_list, 
                         EmbeddingType type                      ) :
            d_seq      (d_seq                               ),
            d_model    (d_model                             ),
            d_heads    (d_heads                             ),
            repetitions(repetitions                         ),
            tokenizer  (dictionary_list, d_model, d_seq,type),
            model      (                                    ),
            lm_head    (this -> tokenizer.embeddings, d_seq ) {
    for(size_t i = 0; i < this -> repetitions; i++ ) {
        this -> model.push_back(TransformerBlock(d_seq, d_model, d_heads) );
    }
};

void Transformer::run( size_t input_index ) {
    Matrix input = this -> tokenizer.input_token_pe[input_index];
    this -> model[0].forward(input);
    for(size_t i = 1; i < this -> repetitions; i++ ) {
        this -> model[i].forward(this -> model[i-1].ln_ffn.normalized_input);
    }
    this -> lm_head.forward(this -> model[this -> repetitions - 1 ].ln_ffn.normalized_input);
};

void Transformer::feed(std::string text ) {
    this -> tokenizer.text_to_input(text);
};

size_t Transformer::predict(Matrix& res ) {
    float max = -100000;
    size_t max_idx = 0;

    size_t last_row_index = res.rows - 1;
    for(size_t i = 0; i < res.cols; i++) {
        float x = res(last_row_index, i);
        if( x > max) {
            max = x;
            max_idx = i;
        }
    }
    return max_idx;
};

size_t Transformer::predict_k(Matrix& res, size_t k ) {
    std::vector<size_t> max_k;

    size_t last_row_index = res.rows - 1;

    std::vector<std::pair<float, int>> indexed_row(res.cols);
    for(int i = 0; i < res.cols; ++i) {
        indexed_row[i] = {res(last_row_index, i), i};
    }

    std::sort(indexed_row.begin(), indexed_row.end(), [](std::pair<float, int> i, std::pair<float, int> j) { return (j<i); });
    std::vector<std::pair<float, int>> top_k;
    for(size_t i = 0; i < k; i++) {
        top_k.push_back(indexed_row[i]);
    }

    float sum = 0;

    for(auto& v: top_k) sum += v.first;
    for(auto& v: top_k) v.first /= sum;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<float> weights;
    
    for(auto& p : top_k) weights.push_back(p.first);

    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    size_t chosen_idx = top_k[dist(gen)].second;
    return chosen_idx;
};

std::string Transformer::word_from_index(size_t index) {
    return this -> tokenizer.words[index];
}
#endif