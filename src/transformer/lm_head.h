#ifndef LM_HEAD_H
#define LM_HEAD_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "../consts.h"
#include "adam.h"

class LMHead {
public:
    Matrix& embeddings, d_embeddings;
    Matrix d_transformer_output;
    Matrix logits_cache, d_logits;
    Matrix probs_cache;
    
    Adam adam;

    std::vector<size_t> max_idx;

    LMHead(Matrix& embeddings, size_t d_seq );

    Matrix& forward(Matrix& transformer_output);

    Matrix& backward(std::vector<size_t>&  target, Matrix& transformer_output);

    void learn();
    void step();
};

#endif