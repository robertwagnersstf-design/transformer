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
    std::cout << "\nMat B(A Transposed) \n\n";

    for(size_t i = 0; i < 3; i++ ) {
        for(size_t j=0; j < 4; j++) {
            mat_b(i,j) = i*j+j;
        }
    }
    mat_b.transpose();
    mat_b.print();
    Matrix mat_c = mat_a * mat_b;

    std::cout << "\nMat A * Mat B \n\n";

    mat_c.print();

    std::cout << "\nMat C + Mat C \n\n";
    mat_c += mat_c;
    mat_c.print();

    float half = 0.5;

    std::cout << "\n0.5*Mat B \n\n";
    mat_b *= half;
    mat_b.print();

    std::cout << "\nMat C - 0.5*Mat C \n\n";
    Matrix mat_d = mat_c * half;
    mat_c -= mat_d;

    mat_c.print();
   
    mat_c += (mat_a * mat_b );

    std::cout << "Mat C + Mat A * Mat B \n\n";
    mat_c.print();


    std::cout << "\nStatic Mat A * Mat B \n\n";
    Matrix mat_e(3,3, false);
    Matrix::gemm(mat_a,mat_b, mat_e);
    mat_e.print();

    !mat_e;

    std::cout << "\nEmpty MatE\n";
    mat_e.print();

    Matrix mat_slice(8, 9, false);
    std::cout << "\nMat Slice \n\n";

    for(size_t i = 0; i < 8; i++ ) {
        for(size_t j=0; j < 9; j++) {
            mat_slice(i,j) = i*j+j;
        }
    }
    mat_slice.print();

    Matrix slice = mat_slice.slice(2,3,3,3);

    std::cout << "\nSlice \n\n";
    slice.print();
    
    std::cout << "\nSlice Softmax\n\n";
    slice.layer_norm();
    slice.print();
    return 0;
}