#include "tokenizer.h"

#ifndef TOKENIZER_CPP
#define TOKENIZER_CPP


Tokenizer::Tokenizer(std::vector<std::string> dictionary_list, size_t embedd_dims, size_t d_model, EmbeddingType type ) : 
                embeddings   (dictionary_list.size() + 3 , embedd_dims, false),
                embedd_dims  (embedd_dims                                    ),
                dict_size    (dictionary_list.size()                         ),
                dictionary   (                                               ),
                d_model      (d_model                                        ),
                input_token  (d_model, embedd_dims                           ),
                embeddingtype(type                                           ) {
    size_t curr_idx = 3;
    this -> dictionary.insert(std::pair<std::string,size_t>("UNK", 0));
    this -> dictionary.insert(std::pair<std::string,size_t>("PAD", 1));
    this -> dictionary.insert(std::pair<std::string,size_t>("#"  , 2));
    
    for(auto word: dictionary_list) {
        std::pair<std::string,size_t> entry(word, curr_idx);
        this -> dictionary.insert(entry);
        ++curr_idx;
    }
    this -> embeddings.embedding_init();
};

Matrix& Tokenizer::text_to_input(const std::string& text ) {
    std::vector<std::string> tokens;
    std::string to_split = text;

    split_to_tokens(to_split, tokens, this -> embeddingtype == EmbeddingType::ByCharacter ? "" : " ");

    size_t current_row = 0;
    for(auto token: tokens ) {
        auto it = this -> dictionary.find(token);
        size_t token_index = 0;
        if (it != this -> dictionary.end() ) {
            token_index = it -> second;
        }
        for(size_t j = 0; j < this -> embedd_dims; j++ ) {
            this -> input_token(current_row, j ) = this -> embeddings(token_index, j);
        } 
        ++current_row;
    }
    if(current_row < this -> d_model ) {
        for(size_t i = current_row; i < this -> d_model ; i++ ) {
            for(size_t j = 0; j < this -> embedd_dims; j++ ) {
                this -> input_token(i, j ) = this -> embeddings(1, j); //Padding
            }
        }
    }
    return this -> input_token;
};

void Tokenizer::split_to_tokens(const std::string& text, std::vector<std::string>& target, const std::string delimiter ) {
    if (delimiter.empty()) {
        for (char c : text) {
            target.push_back(std::string(1,c));
        }
    } else {
        size_t start = 0;
        size_t end   = text.find(delimiter);
        std::string token = text.substr(start, end-start);
        while( end != std::string::npos ) {
            if( end - start > 0 ) {
                target.push_back(text.substr(start, end-start) );
            }

            start = end + delimiter.length();
            end   = text.find(delimiter, start);
        }

        if( start < text.length() ) {
            target.push_back(text.substr(start) );
        }
    }
}


#endif