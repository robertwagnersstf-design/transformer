#include "transformer_block.h"

#ifndef TRANSFORMER_BLOCK_CPP
#define TRANSFORMER_BLOCK_CPP

TransformerBlock::TransformerBlock(size_t d_seq, size_t d_model, size_t d_heads):
    ln_attention(d_seq, d_model                                     ),
    ln_ffn      (d_seq, d_model                                     ),
    mha         (d_seq, d_model, d_heads                            ),
    input       (d_seq, d_model                                     ),
    ffn_input   (d_seq, d_model                                     ),
    grad_mid    (d_seq, d_model                                     ),
    grad_final  (d_seq, d_model                                     ),
    block_output(d_seq, d_model                                     ),
    expansion   (d_model, d_model*EXPANSION_MULTIPLIER, d_seq       ),
    final       (d_model*EXPANSION_MULTIPLIER,d_model , d_seq, false)  {

};

Matrix& TransformerBlock::forward(Matrix& input) {
    // 1. Zweig: Attention
    this->ln_attention.forward(input); // x -> LN(x)
    this->mha.forward_mha(this->ln_attention.normalized_input);
    
    // Residual 1: Originaler Input + MHA Output
    // Das Ergebnis speichern wir in einem Zwischen-Cache
    Matrix::gema(input, this->mha.cache.output_w0, this->ffn_input); 

    // 2. Zweig: FFN
    this->ln_ffn.forward(this->ffn_input); // LN auf das Ergebnis von oben
    this->expansion.forward(this->ln_ffn.normalized_input);
    this->final.forward(this->expansion.act);

    // Residual 2: ffn_input + FFN Output

    Matrix::gema(this->ffn_input, this->final.act, this-> block_output); 
    return this->block_output;
};

Matrix& TransformerBlock::backward(Matrix& gradient) {
    this -> grad_mid.zero_init();
    this -> grad_final.zero_init();
    // --- Teil 2: FFN Zweig ---
    // Der Gradient kommt vom nächsten Block
    this->final.backward(gradient);
    this->expansion.backward(this->final.d_error);
    this->ln_ffn.backward(this->expansion.d_error);
    
    // Der Fehler für den Pfad davor ist: 
    // Gradient (Residual) + Fehler durch das LN/FFN
    Matrix::gema(gradient, this->ln_ffn.d_normalized_input, this->grad_mid);

    // --- Teil 1: Attention Zweig ---
    this->mha.backward_mha(this->grad_mid);
    this->ln_attention.backward(this->mha.cache.d_input);

    // Der finale Gradient für den Block davor:
    // grad_mid (Residual) + Fehler durch LN/Attention
    Matrix::gema(this->grad_mid, this->ln_attention.d_normalized_input, this->grad_final);
    return this->grad_final;
};

void TransformerBlock::learn() {
    this -> ln_ffn.learn();
    this -> final.learn();
    this -> expansion.learn();
    this -> ln_attention.learn();
    this -> mha.learn();
}

void TransformerBlock::step() {
    this -> ln_ffn.step();
    this -> final.step();
    this -> expansion.step();
    this -> ln_attention.step();
    this -> mha.step();
}
#endif