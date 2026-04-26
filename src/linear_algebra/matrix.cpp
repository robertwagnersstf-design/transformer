#include "matrix.h"

 Matrix::Matrix(size_t r, size_t c, bool initialize) {
    this -> rows   = r;
    this -> cols   = c;
    this -> stride = c;
    this -> transposed = false;
    this -> data.resize(r * c, 0.0f);

    if( initialize) {
        this -> xavier_init();
    }
 };

 void Matrix::xavier_init() {
    float limit = sqrt(6.0f / (rows + cols));
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto& val : this -> data) val = dist(gen);
    std::cout <<"Matrix initialized with xavier\n";
 };

 float& Matrix::operator()(size_t r, size_t c) {
    if( this -> transposed ) {
        return this -> data[this -> stride * c + r ];
    }
    return this -> data[this -> stride * r + c ];
 };

 const float& Matrix::operator()(size_t r, size_t c) const {
    if( this -> transposed ) {
        return this -> data[this -> stride * c + r ];
    }
    return this -> data[this -> stride * r + c ];
 };

 const Matrix& Matrix::transpose() {
    this -> transposed = !this -> transposed;
    std::swap(this -> rows, this -> cols);
    std::cout <<"Matrix transposed\n";
    return *this;
 };

 void Matrix::print() {
    for(size_t i = 0; i < this -> rows; i++ ) {
        for(size_t j = 0; j < this -> cols; j++ ) {
            std::cout << (*this)(i, j) <<"\t";
        }
        std::cout << "\n";
    }
 }