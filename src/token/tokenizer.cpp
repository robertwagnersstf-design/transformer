#include "tokenizer.h"

#ifndef TOKENIZER_CPP
#define TOKENIZER_CPP


Tokenizer::Tokenizer(std::vector<std::string> dictionary_list, size_t d_model, size_t d_seq, EmbeddingType type ) : 
                embeddings        (dictionary_list.size() + 3 , d_model, false),
                d_embeddings      (dictionary_list.size() + 3 , d_model, false),
                m_embeddings      (dictionary_list.size() + 3 , d_model, false),
                v_embeddings      (dictionary_list.size() + 3 , d_model, false),
                d_model           (d_model                                    ),
                dict_size         (dictionary_list.size()                     ),
                dictionary        (                                           ),
                d_seq             (d_seq                                      ),
                input_token_pe    (                                           ),
                input_token       (                                           ),
                postional_encoding(d_seq, d_model                             ),
                embeddingtype     (type                                       ),
                sequence          (                                           ),
                words             (                                           ) {
    size_t curr_idx = 3;
    this -> dictionary.insert(std::pair<std::string,size_t>("UNK", 0));
    this -> dictionary.insert(std::pair<std::string,size_t>("PAD", 1));
    this -> dictionary.insert(std::pair<std::string,size_t>("#"  , 2));
    
    words = {"UNK", "PAD", "#"};

    for(auto word: dictionary_list) {
        std::pair<std::string,size_t> entry(word, curr_idx);
        this -> dictionary.insert(entry);
        words.push_back(word);
        ++curr_idx;
    }
    this -> embeddings.embedding_init();
    this -> postional_encoding.positional_encoding_init();

    this -> d_embeddings.zero_init();
    this -> m_embeddings.zero_init();
    this -> v_embeddings.zero_init();
};

Matrix& Tokenizer::text_to_input(const std::string& text ) {
    std::vector<std::string> tokens;
    std::string to_split = text;
    std::vector<size_t> new_sequence;

    split_to_tokens(to_split, tokens, this -> embeddingtype == EmbeddingType::ByCharacter ? "" : " ");

    Matrix new_input(d_seq, d_model);
    Matrix new_pe(d_seq, d_model);

    size_t current_row = 0;
    for(auto token: tokens ) {
        auto it = this -> dictionary.find(token);
        size_t token_index = 0;
        if (it != this -> dictionary.end() ) {
            token_index = it -> second;
        }
        for(size_t j = 0; j < this -> d_model; j++ ) {
            new_input(current_row, j ) = this -> embeddings(token_index, j);
            new_sequence.push_back(token_index);
        } 
        ++current_row;
    }
    if(current_row < this -> d_seq ) {
        for(size_t i = current_row; i < this -> d_seq ; i++ ) {
            for(size_t j = 0; j < this -> d_model; j++ ) {
                new_input(i, j ) = this -> embeddings(1, j); //Padding
                new_sequence.push_back(1);
            }
        }
    }
    this -> input_token.push_back(new_input);
    this -> sequence.push_back(new_sequence);

    Matrix::gema(new_input, this -> postional_encoding, new_pe);

    new_pe *= sqrt(float(this -> d_seq) );
    
    this -> input_token_pe.push_back(new_pe);
    return new_pe;
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