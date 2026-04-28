#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <random>
#include <memory>
#include "../consts.h"
#include <array>
#include <cmath>
#include <iomanip>

#define statistics_block std::vector<std::array<float,2 >> 

class Matrix {
public:
    size_t rows, cols, stride, offset;
    std::shared_ptr<std::vector<float>> data;
    bool transposed;
    
    Matrix(size_t r, size_t c, bool initialize = false);

    // Zugriff via m(row, col)
    float& operator()(size_t r, size_t c);

    const float& operator()(size_t r, size_t c) const;

    const Matrix  operator   *(const Matrix& m );
    const Matrix  operator   *(float k   );
    const Matrix& operator  *=(float k   );
    const Matrix& operator +=(const Matrix& m  );
    const Matrix& operator -=(const Matrix& m  );
    const Matrix& operator ! ();
    const Matrix& transpose();

    const Matrix slice  (const size_t row_start, const size_t col_start, const size_t rows, const size_t cols );
    void deslice(const size_t row_start, const size_t col_start, Matrix& slice );

    void activate_relu( Matrix& m);
    void leaky_relu_backward(Matrix& dX);

    statistics_block layer_norm( Matrix& m);
    void layer_norm( Matrix& m, Matrix & beta, Matrix & gamma, std::vector<float>& means, std::vector<float>& inv_devs);

    void ms_softmax( Matrix& m);
    void ms_softmax( );
    void ms_softmax_backward(const Matrix& dX, Matrix& dY);

    void print(std::string label = "Matrix");   

    void xavier_init();
    void he_init();
    void embedding_init(float sigma = 0.02f);
    void zero_init();
    void value_init(float val);
    void positional_encoding_init( );

    Matrix copy();
    Matrix copy(Matrix& m);
    
    // static stuff
    static void gemm(const Matrix& a, const Matrix& b, Matrix& target ) {
        if ( a.cols != b.rows       || 
            target.rows != a.rows || 
            target.cols != b.cols ) {
            throw std::runtime_error("Dimension mismatch!");
        }

        for(size_t i = 0; i < a.rows; i++ ) {
            for(size_t k = 0; k < b.cols; k++ ) {
                float val_c = 0;
                for(size_t j = 0; j < a.cols; j++ ) {
                    float val_a = a( i , j );
                    float val_b = b(j , k );
                    val_c += val_a * val_b;
                }
                target(i, k) = val_c;
            }
        }
    }
    static void gema(const Matrix& a, const Matrix& b, Matrix& target ) {
        if (a.cols != b.cols ) {
            throw std::runtime_error("Dimension mismatch!");
        }
        // schnelle variante für den standardfall
        if (b.rows == 1 ) {
            for(size_t i = 0; i < a.rows; i++ ) {
                for(size_t j = 0; j < b.cols; j++ ) {
                    target(i,j ) = a(i,j) + b(0,j);
                }
            }
        } else if(a.rows == b.rows ){
            for(size_t i = 0; i < a.rows; i++ ) {
                for(size_t j = 0; j < b.cols; j++ ) {
                    target(i,j ) = a(i,j) + b(i,j);
                }
            }
        } else {
            throw std::runtime_error("Dimension mismatch!");
        }
    };
};

#endif