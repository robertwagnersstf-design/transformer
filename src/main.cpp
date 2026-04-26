#include <iostream>
#include <vector>
#include <numeric>
#include "linear_algebra/matrix.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

int main() {
    std::cout << "--- Transformer Dev Environment ---" << std::endl;
    Matrix mat_a(3, 4, false);
    std::cout << "Mat A \n\n";

    for(size_t i = 0; i < 3; i++ ) {
        for(size_t j=0; j < 4; j++) {
            mat_a(i,j) = i*j+j;
        }
    }
    mat_a.print();
    Matrix mat_b(3, 4, false);
    std::cout << "Mat B(A Transposed) \n\n";

    for(size_t i = 0; i < 3; i++ ) {
        for(size_t j=0; j < 4; j++) {
            mat_b(i,j) = i*j+j;
        }
    }
    mat_b.transpose();
    mat_b.print();
    Matrix mat_c = mat_a * mat_b;

    std::cout << "Mat A * Mat B \n\n";

    mat_c.print();

    std::cout << "Mat C + Mat C \n\n";
    mat_c += mat_c;
    mat_c.print();

    float half = 0.5;

    std::cout << "0.5*Mat B \n\n";
    mat_b *= half;
    mat_b.print();

    std::cout << "Mat C - 0.5*Mat C \n\n";
    Matrix mat_d = mat_c * half;
    mat_c -= mat_d;

    mat_c.print();
   
    mat_c += (mat_a * mat_b );

    std::cout << "Mat C + Mat A * Mat B \n\n";
    mat_c.print();
    return 0;
}