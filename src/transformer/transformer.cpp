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
            model      (                                    ) {
    for(size_t i = 0; i < this -> repetitions; i++ ) {
        this -> model.push_back(TransformerBlock(d_seq, d_model, d_heads) );
    }
};

void Transformer::teach(std::string text) {
    Matrix input = this -> tokenizer.text_to_input(text);
}
#endif