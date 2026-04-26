#include <iostream>
#include <vector>
#include <numeric>
#include "linear_algebra/matrix.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

int main() {
    std::cout << "--- Transformer Dev Environment ---" << std::endl;
    Matrix* mat_a = new Matrix(3, 4, false);
    std::cout << "Mat A \n\n";
    for(size_t i = 0; i < 3; i++ ) {
        for(size_t j=0; j < 4; j++) {
            mat_a -> set(i,j, i*j+j);
        }
    }
    mat_a -> print();
    Matrix* mat_b = new Matrix(3, 4, false);
    std::cout << "Mat A \n\n";
    for(size_t i = 0; i < 3; i++ ) {
        for(size_t j=0; j < 4; j++) {
            mat_b -> set(i,j, i*j+j);
        }
    }
    mat_b -> transpose();
    mat_b -> print();
    Matrix mat_c = *mat_a * *mat_b;

    std::cout << "Mat A * Mat B \n\n";
    
    mat_c.print();

    delete mat_a;
    delete mat_b;

    return 0;
}