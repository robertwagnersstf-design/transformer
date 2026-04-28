#include <iostream>
#include <vector>
#include <numeric>
#include "linear_algebra/matrix.h"
#include "token/tokenizer.h"
#include "transformer/multi_head_attention_layer.h"

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

    Matrix mat_target(3, 4, false);
    std::cout << "\nStatic MatA + MatA\n\n";
    Matrix::gema(mat_a,mat_a, mat_target);
    mat_target.print();

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
    
    Matrix mat_softmax_ex(8, 9, false);
    std::vector<std::array<float,2 >> k = mat_slice.layer_norm(mat_softmax_ex);

    std::cout << "\nSlice Layernorm not inline\n\n";
    mat_softmax_ex.print();

    for( auto v: k ) 
        std::cout << "\nMean: " << v[0] << ", Variance: " << v[1] <<"\n";

    Matrix mat_softmax(4, 4, false);
    Matrix dx(4, 4, false);
    Matrix dy(4, 4, false);

    for(size_t i = 0; i < 4; i++ ) {
        for(size_t j=0; j < 4; j++) {
            mat_softmax(i,j) = float(i+j)/8.;
        }
    }

    std::cout << "\n\nMatrix to do Softmax\n\n";

    mat_softmax.print();
    Matrix mat_forward(4, 4, false);
    mat_softmax.ms_softmax(mat_forward);

    std::cout << "\n\nSoftmax Result\n\n";

    mat_forward.print();

    for(size_t i = 0; i < 4; i++ ) {
        for(size_t j=0; j < 4; j++) {
            dx(i,j) = i==j ? .5 : 1.;
        }
    }
    
    std::cout << "\n\nGradient dX\n\n";

    dx.print();

    mat_forward.ms_softmax_backward(dx, dy);

    std::cout << "\n\nResult dY\n\n";
    dy.print();

    Matrix mat_embedd(6, 10, false);
    mat_embedd.embedding_init();
    Matrix mat_pe(6, 10, false);
    mat_pe.positional_encoding_init();

    std::cout << "\n\nEmbedding init\n\n";
    mat_embedd.print();

    std::cout << "\n\nPE init\n\n";
    mat_pe.print();

    std::cout << "\n\nJust for fun embedd + PE \n\n";
    Matrix mat_add(6, 10, false);

    Matrix::gema(mat_embedd, mat_pe, mat_add);
    mat_add.print();
    
    std::cout <<"\n\n Token tests\n\n";

    std::vector<std::string> tokenres = std::vector<std::string>();

    std::cout <<"\n\n Split by word\n\n";

    Tokenizer tokenizer({"klaus", "kann", "lullen", "ohne", "dass", "es", "brennt", "mag", "darf", "will", "bis"}, 6, 8);
    tokenizer.split_to_tokens("klaus will lullen, bis es brennt", tokenres, " ");
    
    for(auto token: tokenres) {
        std::cout << token <<"    ";
    }

    tokenres.clear();
    
    tokenizer.text_to_input("klaus will lullen bis  es brennt");
    std::cout <<"\nInputmebdding\n";
    tokenizer.input_token[0].print();

    MultiHeadAttention m(8, 6, 3);
    Matrix input(8,6, false);

    for(size_t i = 0; i < 8; i++ ) {
        for(size_t j=0; j < 6; j++) {
            input(i,j) = i+j;
        }
    }

    m.forward_mha(input);
    std::cout <<"\n Input Matrix \n";
    input.print();

    std::cout <<"\n WQ Matrix \n";
    m.w_q.print();
    std::cout <<"\n WK Matrix \n";
    m.w_k.print();
    std::cout <<"\n WV Matrix \n";
    m.w_v.print();
    std::cout <<"\n W0 Matrix \n";
    m.w0.print();
    std::cout <<"\n Q Matrix \n";
    m.cache.q.print();
    std::cout <<"\n K Matrix \n";
    m.cache.k.print();
    std::cout <<"\n V Matrix \n";
    m.cache.v.print();
    m.cache.q.print();
    std::cout <<"\n Score Matrix 1 \n";
    m.cache.score_heads[0].print();
    std::cout <<"\n Score Matrix 2 \n";
    m.cache.score_heads[1].print();
    std::cout <<"\n Score Matrix 3 \n";
    m.cache.score_heads[2].print();
    std::cout <<"\n output Matrix \n";
    m.cache.output.print();
    std::cout <<"\n output Matrix W0 \n";
    m.cache.output_w0.print();
   
    MultiHeadAttention m2(8, 6, 3);
    std::cout <<"\n All weigths to 1 \n";

    std::fill(m2.w_q.data -> begin(), m2.w_q.data -> end(), 1.);
    std::fill(m2.w_k.data -> begin(), m2.w_k.data -> end(), 1.);
    std::fill(m2.w_v.data -> begin(), m2.w_v.data -> end(), 1.);
    std::fill(m2.w0.data -> begin(),  m2.w0.data  -> end(), 1.);

    m2.forward_mha(input);
    std::cout <<"\n Input Matrix \n";
    input.print();

    std::cout <<"\n WQ Matrix \n";
    m2.w_q.print();
    std::cout <<"\n WK Matrix \n";
    m2.w_k.print();
    std::cout <<"\n WV Matrix \n";
    m2.w_v.print();
    std::cout <<"\n W0 Matrix \n";
    m2.w0.print();
    std::cout <<"\n Q Matrix \n";
    m2.cache.q.print();
    std::cout <<"\n K Matrix \n";
    m2.cache.k.print();
    std::cout <<"\n V Matrix \n";
    m2.cache.v.print();

    std::cout <<"\n Score Matrix 1 \n";
    m2.cache.score_heads[0].print();
    std::cout <<"\n Score Matrix 2 \n";
    m2.cache.score_heads[1].print();
    std::cout <<"\n Score Matrix 3 \n";
    m2.cache.score_heads[2].print();
    std::cout <<"\n output Matrix \n";
    m2.cache.output.print();
    std::cout <<"\n output Matrix W0 \n";
    m2.cache.output_w0.print();
    return 0;
}