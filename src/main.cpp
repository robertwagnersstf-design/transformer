#include <iostream>
#include <vector>
#include <numeric>
#include "linear_algebra/matrix.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

int main() {
    std::cout << "--- Transformer Dev Environment ---" << std::endl;
    Matrix* mat = new Matrix(3, 4, true);
    mat -> print();
    mat -> transpose();
    mat -> print();
    delete mat;
    return 0;
}