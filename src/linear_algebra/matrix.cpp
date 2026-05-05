#include "matrix.h"

#ifndef MATRIX_CPP
#define MATRIX_CPP

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

 void Matrix::deslice(const size_t row_start, const size_t col_start, Matrix& slice ) {
   for(size_t i = row_start; i < row_start + slice.rows; i++ ) {
      for(size_t j = col_start; j < col_start + slice.cols; j++ ) {
         (*this)(i,j) = slice(i-row_start, j-col_start);
      }
   }
 };

 void Matrix::xavier_init() {
    static std::mt19937 gen(std::random_device{}()); 
    
    float n_in = static_cast<float>(this->rows);
    float n_out = static_cast<float>(this->cols);
    float limit = std::sqrt(6.0f / (n_in + n_out));
    
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (size_t i = 0; i < this -> rows; i++) {
        for (size_t j = 0; j < this -> cols; j++) {
            (*this)(i, j) = dist(gen);
        }
    }
 };

 void Matrix::col_sums(Matrix& m) {
   for(size_t j = 0; j < this -> cols; j++ ) {
      float x = 0.f;
      for(size_t i = 0; i < this -> rows; i++ ) {
         x += (*this)(i,j);
      }
      m(0,j) = x;
   }
 }

 void Matrix::he_init() {
    // Bei He-Init ist oft nur die Anzahl der Eingänge (rows bei Gewichtsmatrix) entscheidend
    float limit = std::sqrt(6.0f / this -> rows); 
    static std::mt19937 gen(std::random_device{}()); 
    std::uniform_real_distribution<float> dist(-limit, limit);
    
    for (size_t i = 0; i < this -> rows; i++) {
        for (size_t j = 0; j < this -> cols; j++) {
            (*this)(i, j) = dist(gen);
        }
    }
}

void Matrix::zero_init() {
   std::fill(this -> data ->begin(), this -> data -> end(), 0.);
};

void Matrix::value_init(float val) {
   std::fill(this -> data ->begin(), this -> data -> end(), val);
};

 void Matrix::embedding_init(float sigma) {
    static std::mt19937 gen(std::random_device{}()); 
    std::normal_distribution<float> distribution(0.0, sigma);

    for (size_t i = 0; i < rows * cols; ++i) {
        (*this -> data)[i] = distribution(gen);
    }
}

 void Matrix::positional_encoding_init( ) {
    for(size_t i = 0; i < this -> rows; i++ ) {
      for(size_t j = 0; j < this -> cols; j++ ) { 
         float x = 0.;
         float dimension_index = float(j / 2);
         float div_term = pow(10000.f, (2.*dimension_index/float(this -> cols) ) );

         if(j%2 == 0 ) x = std::sin(float(i)/ div_term );
         else          x = std::cos(float(i)/ div_term );
         (*this)(i,j) = x;
      }   
   } 
 }

void Matrix::activate_relu(Matrix& m) {
   for(size_t i = 0; i < this -> rows; i++ ) {
      for(size_t j = 0; j < this -> cols; j++ ) { 
         float x = (*this)(i,j);
         m(i,j) = x >= 0 ? x : RELU_LEAKAGE * x;
      }   
   } 
 }

void Matrix::leaky_relu_backward(Matrix& dX) {
    for (size_t i = 0; i < this -> rows; ++i) {
        for (size_t j = 0; j < this -> cols; ++j) {
            float val = (*this)(i, j);
            dX(i, j) = (val > 0) ? 1.0f : RELU_LEAKAGE;
        }
    }
}

void Matrix::layer_norm( Matrix& m, Matrix & beta, Matrix & gamma, std::vector<float>& means, std::vector<float>& inv_devs) {
   
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
            m(i, j) = ( ((*this)(i, j) - mean) * inv_std ) * gamma(0,j) + beta(0,j);
         }
         means[i]    = mean, 
         inv_devs[i] = inv_std;
      }
};

void Matrix::layer_norm_backward( Matrix& g, Matrix& m, std::vector<float>& means, std::vector<float>& inv_devs, Matrix & gamma) {
   for(size_t i = 0; i < this -> rows; i++ ) {
      float sum_mean = 0;
      float sum_std  = 0;
      for(size_t j = 0; j < this -> cols;  j++ ) {
         sum_mean += g(i,j);
         sum_std  += g(i,j) * (*this)(i,j);
      }
      for(size_t j = 0; j < this -> cols;  j++ ) {
         m(i,j) = ( gamma(0,j) * inv_devs[i]/ this -> cols ) * (this -> cols * g(i,j) - sum_mean - (*this)(i,j) * sum_std);
      }
   }
};

void Matrix::ms_softmax(Matrix& m) {
    for (size_t i = 0; i < this -> rows; ++i) {
        // 1. Mittelwert (Mean) berechnen
        float max = 0.0f;
        for (size_t j = 0; j < this -> cols; ++j) {
            max = (*this)(i, j) > max ? (*this)(i, j) : max;
        }
        
        float sum_exp = 0.0f;
        for (size_t j = 0; j < this -> cols; ++j) {
            float x = std::exp((*this)(i, j) - max);
            m(i, j) = x;
            sum_exp += x;
        }
        
        for (size_t j = 0; j < this -> cols; ++j) {
            m(i, j) /= sum_exp;
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

void Matrix::ms_softmax_backward(const Matrix& dX, Matrix& dY) {
      for (size_t i = 0; i < this -> rows; i++) {
          float dot = 0.f;
         for (size_t j = 0; j < this -> cols; j++) {
            dot += (*this)(i,j) * dX(i,j);
         }
         float res = 0.f;
         for(size_t k = 0; k < this -> cols; k++ ) {
            dY(i,k) = (*this)(i,k) * (dX(i,k) - dot); 
         }
      }
  };

void Matrix::clip_gradients() {
    float sum_sq = 0.0f;
    // 1. Berechne die L2-Norm (Euklidische Länge)
    for (size_t i = 0; i < rows * cols; i++) {
        sum_sq += (*this -> data)[i] * (*this -> data)[i];
    }
    // 2. Wenn die Norm zu groß ist, skaliere alles runter
    if (sum_sq > GRADIENT_THRESHOLD) {
        float norm = std::sqrt(sum_sq);
        float scale_factor = GRADIENT_THRESHOLD / norm;
        for (size_t i = 0; i < rows * cols; i++) {
            (*this -> data)[i] *= scale_factor;
        }
    }
};

 Matrix Matrix::copy() {
   Matrix m( this -> rows, this -> cols );
   m.transposed = this -> transposed;
   m.offset     = this -> offset;
   m.stride     = this -> stride;
   m.data       = std::make_shared<std::vector<float>>(*this->data);

   return m;
 }

 //copy in already existing matrix of equal dimensions, to aovid unnesecary initialization
 Matrix Matrix::copy(Matrix& m) {
   if (this->cols != m.cols  || this -> rows != m.rows ) {
      throw std::runtime_error("Dimension mismatch!");
   }

   m.transposed = this -> transposed;
   m.offset     = this -> offset;
   m.stride     = this -> stride;
   for(size_t i = 0; i < this -> rows; i++ ) {
      for(size_t j = 0; j < this -> cols; j++) {
         m(i,j) = (*this)(i,j);
      }
   }

   return m;
 }
 std::vector<float> Matrix::get_row(size_t row) {
   std::vector<float> out;
   for(size_t j = 0; j < this -> cols; j++ ) {
      out.push_back((*this)(row,j ));
   }
   return out;
 };

 void Matrix::set_row(size_t row, float value) {
   for(size_t j = 0; j < this -> cols; j++ ) {
      (*this)(row,j ) = value;
   }
 };
 
 void Matrix::print(std::string label) {
    std::cout << "--- " << label << " ---" << std::endl;
    // Festlegen: 2 Nachkommastellen, feste Breite von 10 Zeichen
    std::cout << std::fixed << std::setprecision(9); 
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // std::setw(10) sorgt dafür, dass jede Zahl exakt 10 Zeichen Platz braucht
            std::cout << std::setw(10) << (*this)(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
 }

 #endif