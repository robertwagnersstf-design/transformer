#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "../consts.h"

#define dict_type std::unordered_map<std::string, int>

class Tokenizer {
public:
    Matrix embeddings;
    Matrix input_token;
    const size_t dict_size, embedd_dims, d_model;
    dict_type dictionary;
    EmbeddingType embeddingtype;

    Tokenizer(std::vector<std::string> dictionary_list, size_t embedd_dims, size_t d_model, EmbeddingType type = EmbeddingType::ByWord);
    Matrix& text_to_input(const std::string& text );
    void split_to_tokens(const std::string& text, std::vector<std::string>& target, const std::string delimiter );
};


#endif