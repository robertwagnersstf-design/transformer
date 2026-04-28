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
    Matrix embeddings, d_embeddings, m_embeddings, v_embeddings;
    Matrix postional_encoding;
    std::vector<Matrix> input_token, input_token_pe;
    
    const size_t dict_size, d_model, d_seq;
    dict_type dictionary;
    EmbeddingType embeddingtype;

    Tokenizer(std::vector<std::string> dictionary_list, size_t d_model, size_t d_seq, EmbeddingType type = EmbeddingType::ByWord);
    Matrix& text_to_input(const std::string& text );
    void split_to_tokens(const std::string& text, std::vector<std::string>& target, const std::string delimiter );
};


#endif