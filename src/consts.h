#ifndef CONST_H
#define CONST_H
static const float  RELU_LEAKAGE = 0.001;
static const size_t EXPANSION_MULTIPLIER = 4;

enum EmbeddingType {
  ByCharacter,
  ByWord
}; 

#endif