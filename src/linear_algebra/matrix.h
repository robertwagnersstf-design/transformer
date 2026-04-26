#include <vector>
#include <iostream>
#include <random>
#include <memory>
#include "../consts.h"

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

    const Matrix slice(const size_t row_start, const size_t col_start, const size_t rows, const size_t cols );

    void activate_relu();
    void layer_norm();
    void ms_softmax();

    void print();   
    void xavier_init();
    void he_init();

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
};