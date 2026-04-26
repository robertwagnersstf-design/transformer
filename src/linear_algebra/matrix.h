#include <vector>
#include <iostream>
#include <random>

class Matrix {
public:
    size_t rows, cols, stride;
    std::vector<float> data;
    bool transposed;

    Matrix(size_t r, size_t c, bool initialize = false);

    // Zugriff via m(row, col)
    float& operator()(size_t r, size_t c);

    const float& operator()(size_t r, size_t c) const;

    const Matrix operator   *(Matrix& m );
    const Matrix& operator +=(Matrix& m );
    const Matrix& operator -=(Matrix& m );
    const Matrix& operator ! ();
    const Matrix& transpose();
    const Matrix& set(size_t r, size_t c, float val);
    
    void print();
    void xavier_init();
};