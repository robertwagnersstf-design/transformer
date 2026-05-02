#ifndef ADAM_H
#define ADAM_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include "../linear_algebra/matrix.h"
#include "../consts.h"
#include <math.h> 

class Adam {
public:
    Matrix m, v;
    int t = 0;
    Adam(size_t r, size_t c);
    void step(Matrix& d_w);
    void learn(Matrix& w);
};

#endif