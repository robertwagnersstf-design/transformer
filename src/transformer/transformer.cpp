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
        this -> model[i].forward(this -> model[i-1].block_output);
    }
    this -> lm_head.forward(this -> model[this -> repetitions - 1 ].block_output);
};

void Transformer::backprop(size_t input_index ) {
    this -> lm_head.backward(this -> tokenizer.sequence[input_index], this -> model[this -> repetitions - 1].block_output);
    this -> model[this -> repetitions - 1].backward(this -> lm_head.d_transformer_output);

    for(int i = this -> repetitions - 2; i >= 0; i-- ) {
        this -> model[i].backward(this -> model[i + 1].grad_final);
    }
    this -> tokenizer.backwards(this -> model[0].grad_final, input_index);
};

float Transformer::calc_loss(std::vector<size_t>&  target) {
    float sum_ln = 0.f;
    for(size_t i = 0; i < target.size() - 1; i++ ) {
        size_t token_index = target[i + 1];
        float val = this -> lm_head.probs_cache(i, token_index);
        val = val < 1e-9 ? 1e-9 : val;
        sum_ln -= log( val );
    }
    sum_ln /= float(target.size() - 1);

    std::cout <<"\n-------  Loss of Run " << this -> run_count << ": " << sum_ln <<" --------\n\n";

    return sum_ln;
};

void Transformer::learn() {
    this -> lm_head.learn();
    for(int i = this -> repetitions - 1; i >= 0; i-- ) {
        this -> model[i].learn();
    }
    this -> tokenizer.learn();
};

void Transformer::step() {
    this -> lm_head.step();
    for(int i = this -> repetitions - 1; i >= 0; i-- ) {
        this -> model[i].step();
    }
    this -> tokenizer.step();
};


void Transformer::training_loop(const std::vector<std::string> & data) {
    size_t amnt = data.size();
    for(auto t: data) {
        this -> tokenizer.text_to_input(t);
    }
    this -> run_count = 0;
    size_t run_count_last_loss_impr = 0;
    float best_loss = 1e7;
    bool run = true;
    size_t curr_run = 0;
    while(run) {
        this -> run(curr_run);
        this -> backprop(curr_run);
        this -> step();
        this -> learn();
        //this -> model[0].mha.d_q.print("D_Q");
        //this -> model[0].ln_attention.d_beta.print("Attention D_BETA");
        //this -> model[0].ln_attention.d_gamma.print("Attention D_GAMMA");
        //this -> model[0].expansion.d_error.print("D Error Expansion");
        //this -> model[0].final.d_error.print("D Error Final");
        //this -> model[0].ln_ffn.d_beta.print("FFN D_BETA");
        //this -> model[0].ln_ffn.d_gamma.print("FFN D_GAMMA");
        //this -> lm_head.d_embeddings.print("D_EMBEDDINGS"); 
        float loss = this -> calc_loss(this -> tokenizer.sequence[curr_run]);
        if( loss < best_loss ) {
            best_loss = loss;
            run_count_last_loss_impr = 0;
        } else {
            run_count_last_loss_impr ++;
        }
        if(loss < .2f ) {
            run = false;
        }

        curr_run = (curr_run + 1) % amnt;
        this -> run_count++;
    }
    for(size_t i = 0; i < this -> tokenizer.sequence.size(); i++ ) {
        this -> run(i);
        this -> test_result(this -> lm_head.probs_cache);
    }
}

void Transformer::feed(std::string text ) {
    this -> tokenizer.text_to_input(text);
};

std::string Transformer::predict_word(std::string text){
    this -> tokenizer.text_to_input(text);
    size_t new_indx = tokenizer.padding_start_index.size() - 1 ;
    this -> run(new_indx);

    // Wir suchen das Ende des tatsächlichen Textes im Buffer
    int last_valid_pos = -1;
    for (size_t j = 0; j < tokenizer.sequence[new_indx].size(); ++j) {
        size_t token = tokenizer.sequence[new_indx][j];
        // 0 = UNK, 1 = PAD, 2 = # -> wir suchen das letzte Wort davor
        if (token != 0 && token != 1 && token != 2) {
            last_valid_pos = j;
        } else {
            break; // Erstes Padding/Stop erreicht
        }
    }
    this -> lm_head.probs_cache.print("Probs for " + text);
    size_t next_token_idx = this->predict(this->lm_head.probs_cache, last_valid_pos);
    return this->word_from_index(next_token_idx);
}

void Transformer::test_result(Matrix& r) {
     std::cout << "Dictionary: \n";

    for(auto n: this -> tokenizer.dictionary) {
        std::cout << n.first  << ": " << n.second <<"; ";
    }

    std::cout << "\n";
    r.print("Props");

    std::cout << "Prediction: Und ";
    for(size_t i = 0; i < r.rows; i++ ) {
        size_t token_index = this -> predict(r,i);
        std::string word = this -> word_from_index(token_index);
        if(word == "#") return;
        std::cout << word << " ";
    }
    std::cout << "\n";
}

size_t Transformer::predict(Matrix& res, size_t row_idx ) {
    float max = -100000;
    size_t max_idx = 0;

    row_idx = row_idx < 0 ? res.rows - 1 : row_idx;
    for(size_t i = 0; i < res.cols; i++) {
        float x = res(row_idx, i);
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