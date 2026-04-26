#include <iostream>
#include <vector>
#include <numeric>
#ifdef _OPENMP
    #include <omp.h>
#endif

int main() {
    std::cout << "--- Transformer Dev Environment ---" << std::endl;
    
    // Teste Vektor (STL)
    std::vector<float> test_matrix(10, 0.5f);
    float sum = std::accumulate(test_matrix.begin(), test_matrix.end(), 0.0f);
    
    std::cout << "STL Check: Summe eines 10er Vektors (0.5): " << sum << std::endl;

    // Teste OpenMP (Multithreading)
    #ifdef _OPENMP
        std::cout << "OpenMP Check: Threads verfuegbar: " << omp_get_max_threads() << std::endl;
    #else
        std::cout << "OpenMP nicht aktiviert." << std::endl;
    #endif

    std::cout << "Setup läuft! Zeit für Goethe." << std::endl;
    
    return 0;
}