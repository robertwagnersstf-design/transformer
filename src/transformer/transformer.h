#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "transformer_block.h"
#include "../token/tokenizer.h"
#include "../consts.h"

class Transformer {
public:
    Transformer(size_t d_seq, size_t d_model, size_t d_heads, size_t repetitions, std::vector<std::string> dictionary_list, EmbeddingType type = EmbeddingType::ByWord);

    size_t d_seq, d_model, d_heads, repetitions;

    Tokenizer tokenizer;
    std::vector<TransformerBlock> model;
    
    void teach(std::string text);
};

#endif