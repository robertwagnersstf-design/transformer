#ifndef CONST_H
#define CONST_H
static const float  RELU_LEAKAGE = 0.01;
static const size_t EXPANSION_MULTIPLIER = 4;
static const float ALPHA   = 0.00005;
static const float BETA1   = 0.9;
static const float BETA2   = 0.999;
static const float EPSILON = 1e-7;
static const int   BATCHSIZE = 50;
static const int   LOSS_CTR  = 3;
static const size_t MAX_LOSS_NO_IMPR=20000;
static const float GRADIENT_THRESHOLD = 1.f;

enum EmbeddingType {
  ByCharacter,
  ByWord
}; 

#endif