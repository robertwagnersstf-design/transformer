#ifndef CONST_H
#define CONST_H
static const float  RELU_LEAKAGE = 0.001;
static const size_t EXPANSION_MULTIPLIER = 4;
static const float ALPHA   = 0.0001;
static const float BETA1   = 0.9;
static const float BETA2   = 0.999;
static const float EPSILON = 1e-7;
static const int   BATCHSIZE = 50;
static const int   LOSS_CTR  = 3;

enum EmbeddingType {
  ByCharacter,
  ByWord
}; 

#endif