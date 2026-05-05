#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "transformer_block.h"
#include "../token/tokenizer.h"
#include "../consts.h"
#include "lm_head.h"

class Transformer {
public:
    Transformer(size_t d_seq, size_t d_model, size_t d_heads, size_t repetitions, std::vector<std::string> dictionary_list, EmbeddingType type = EmbeddingType::ByWord);

    size_t d_seq, d_model, d_heads, repetitions;
    size_t run_count = 0;

    Tokenizer tokenizer;
    std::vector<TransformerBlock> model;
    LMHead lm_head;

    float current_loss = 0.f;
    
    void feed(std::string text);
    void run(size_t input_index);
    void backprop(size_t run_index);
    
    void learn();
    void step();

    void training_loop(const std::vector<std::string>& data);

    size_t predict(Matrix& res, size_t row_idx = -1);
    size_t predict_k(Matrix& res, size_t k );

    std::string word_from_index(size_t index);

    float calc_loss(std::vector<size_t>&  target);
    void test_result(Matrix& r);
    std::string predict_word(std::string text);
    
    void squared_gradient_sum();
    void apply_scale(float f);
};

#endif