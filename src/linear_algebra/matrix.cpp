#include "matrix.h"

 Matrix::Matrix(size_t r, size_t c, bool initialize) {
    this -> rows   = r;
    this -> cols   = c;
    this -> stride = c;

    this -> offset = 0;
    this -> transposed = false;
    this->data = std::make_shared<std::vector<float>>();
    this -> data -> resize(r * c, 0.0f);

    if( initialize) {
        this -> xavier_init();
    }
 };

 float& Matrix::operator()(size_t r, size_t c) {
    if( this -> transposed ) {
        return (*this -> data)[this -> stride * c + r + this -> offset ];
    }
    return (*this -> data)[this -> stride * r + c  + this -> offset ];
 };

 const float& Matrix::operator()(size_t r, size_t c) const {
    if( this -> transposed ) {
        return (*this -> data)[this -> stride * c + r + this -> offset];
    }
    return (*this -> data)[this -> stride * r + c + this -> offset];
 };
 
const Matrix Matrix::operator *(const Matrix& m ) {
   if (this->cols != m.rows) {
      throw std::runtime_error("Dimension mismatch!");
   }

   Matrix target(this -> rows, m.cols, false);
   for(size_t i = 0; i < this -> rows; i++ ) {
      for(size_t k = 0; k < m.cols; k++ ) {
         float val_c = 0;
         for(size_t j = 0; j < this -> cols; j++ ) {
            float val_a = (*this)( i , j );
            float val_b = m(j , k );
            val_c += val_a * val_b;
         }
         target(i, k) = val_c;
      }
   }
   return target;
};

const Matrix Matrix::operator *(float k ) {
   Matrix target(this -> rows, this ->cols, false);
   for(size_t i = 0; i < this -> rows; i++ ) {
      for(size_t j = 0; j < this -> cols; j++ ) { 
         target(i,j) = k * (*this)(i,j);
      }         
   }
   return target;
};

const Matrix& Matrix::operator +=(const Matrix& m ) {
   if (this->cols != m.cols ) {
      throw std::runtime_error("Dimension mismatch!");
   }
   // schnelle variante für den standardfall
  if (m.rows == 1 ) {
      for(size_t i = 0; i < this -> rows; i++ ) {
         for(size_t j = 0; j < this -> cols; j++ ) {
            (*this)(i,j ) += m(0,j);
         }
      }
  } else if(m.rows == this -> rows ){
   for(size_t i = 0; i < this -> rows; i++ ) {
         for(size_t j = 0; j < this -> cols; j++ ) {
            (*this)(i,j ) += m(i,j);
         }
      }
   } else {
      throw std::runtime_error("Dimension mismatch!");
   }
   return *this;
};

const Matrix& Matrix::operator -=(const Matrix& m ) {
   if (this->cols != m.cols  || this -> rows != m.rows ) {
      throw std::runtime_error("Dimension mismatch!");
   }
  for(size_t i = 0; i < this -> rows; i++ ) {
      for(size_t j = 0; j < this -> cols; j++ ) {
         (*this)(i,j ) -= m(i,j);
      }
  }
   return *this;
};

const Matrix& Matrix::operator !() {
   for(size_t i = 0; i < this -> rows; i++ ) {
      for(size_t j = 0; j < this -> cols; j++ ) {
         (*this)(i,j ) = 0.f;
      }
  }
   return *this;
};

const Matrix& Matrix::operator *= (float k ) {
  for(size_t i = 0; i < this -> rows; i++ ) {
      for(size_t j = 0; j < this -> cols; j++ ) { 
         (*this)(i,j) *= k;
      }     
   }
   return *this;
};

 const Matrix& Matrix::transpose() {
    this -> transposed = !this -> transposed;
    std::swap(this -> rows, this -> cols);
    return *this;
 };

 const Matrix Matrix::slice(const size_t row_start, const size_t col_start, const size_t rows, const size_t cols ) {
   size_t offset = this -> stride * row_start + col_start;
   Matrix slice(rows, cols, false);
   slice.data = this -> data;
   slice.offset = offset;
   slice.stride = this -> stride;
   slice.transposed = this -> transposed;
   return slice;
 }

 void Matrix::xavier_init() {
    float limit = sqrt(6.0f / (rows + cols));
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto& val : (*this -> data)) val = dist(gen);
 };


 void Matrix::activate_relu() {
   for(size_t i = 0; i < this -> rows; i++ ) {
      for(size_t j = 0; j < this -> cols; j++ ) { 
         float x = (*this)(i,j);
         (*this)(i,j) = x >= 0 ? x : RELU_LEAKAGE * x;
      }   
   } 
 }

 void Matrix::layer_norm() {
    for (size_t i = 0; i < this -> rows; ++i) {
        // 1. Mittelwert (Mean) berechnen
        float mean = 0.0f;
        for (size_t j = 0; j < this -> cols; ++j) {
            mean += (*this)(i, j);
        }
        mean /= this -> cols;

        // 2. Varianz berechnen
        float variance = 0.0f;
        for (size_t j = 0; j < this -> cols; ++j) {
            float diff = (*this)(i, j) - mean;
            variance += diff * diff;
        }
        variance /= this -> cols;

        // 3. Normalisieren
        float inv_std = 1.0f / std::sqrt(variance + 1e-5f);
        for (size_t j = 0; j < this -> cols; ++j) {
            (*this)(i, j) = ((*this)(i, j) - mean) * inv_std;
         }
      }
  };

  void Matrix::ms_softmax() {
    for (size_t i = 0; i < this -> rows; ++i) {
        // 1. Mittelwert (Mean) berechnen
        float max = 0.0f;
        for (size_t j = 0; j < this -> cols; ++j) {
            max = (*this)(i, j) > max ? (*this)(i, j) : max;
        }
        
        float sum_exp = 0.0f;
        for (size_t j = 0; j < this -> cols; ++j) {
            float x = std::exp((*this)(i, j) - max);
            (*this)(i, j) = x;
            sum_exp += x;
        }
        
        for (size_t j = 0; j < this -> cols; ++j) {
            (*this)(i, j) /= sum_exp;
         }
      }
  };

 void Matrix::print() {
    for(size_t i = 0; i < this -> rows; i++ ) {
        for(size_t j = 0; j < this -> cols; j++ ) {
            std::cout << (*this)(i, j) <<"\t";
        }
        std::cout << "\n";
    }
 }