#include "transformer_block.h"

#ifndef TRANSFORMER_BLOCK_CPP
#define TRANSFORMER_BLOCK_CPP

TransformerBlock::TransformerBlock(size_t d_seq, size_t d_model, size_t d_heads):
    ln_attention(d_seq, d_model                                     ),
    ln_ffn      (d_seq, d_model                                     ),
    mha         (d_seq, d_model, d_heads                            ),
    input       (d_seq, d_model                                     ),
    expansion   (d_model, d_model*EXPANSION_MULTIPLIER, d_seq       ),
    final       (d_model*EXPANSION_MULTIPLIER,d_model , d_seq, false)  {

};

Matrix& TransformerBlock::forward(Matrix& input) {
    //store o copy of the input for later, just to be sure
    //check later whether that might be a reference or pointer
    input.copy(this -> input);
    // calculate attention with d heads
    this -> mha.forward_mha(input);
    //residual connection
    this -> mha.cache.output_w0 += this -> input;
    //normalization
    this -> ln_attention.forward(this -> mha.cache.output_w0);

    this -> expansion.forward(this ->ln_attention.normalized_input);
    this -> final.forward(this -> expansion.act);

    this -> final.act += this -> ln_attention.normalized_input;
    this -> ln_ffn.forward(this -> final.act);

    return this -> ln_ffn.normalized_input;
};

Matrix& TransformerBlock::backward(Matrix& gradient) {
    this -> ln_ffn.backward(gradient);

    this -> final.backward(this -> ln_ffn.d_normalized_input);

    this -> expansion.backward(this -> final.d_error);
    this -> expansion.d_error += gradient;

    this -> ln_attention.backward(this -> expansion.d_error);

    this -> mha.backward_mha(this -> ln_attention.d_normalized_input);
    this -> mha.cache.d_input += this -> expansion.d_error;

    return this -> mha.cache.d_input;
};

void TransformerBlock::learn() {
    this -> ln_ffn.learn();
    this -> final.learn();
    this -> expansion.learn();
    this -> ln_attention.learn();
    this -> mha.learn();
}
#endif