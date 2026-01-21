/**
 * MatMul Test Code for VEGA Compiler Backend Verification
 * 
 * This file contains matrix multiplication implementations to test
 * the correctness of VEGA-generated compiler backends.
 * 
 * Targets: RISC-V, RI5CY (PULP), xCORE
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>

// Configuration
#define MATRIX_SIZE 64
#define EPSILON 1e-5

/**
 * Basic Matrix Multiplication (Naive O(n^3) implementation)
 * Reference implementation for correctness verification
 */
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * Tiled Matrix Multiplication (Cache-friendly version)
 * Better for testing compiler optimization capabilities
 */
void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K, int tile_size = 16) {
    // Initialize C to zero
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }
    
    for (int i0 = 0; i0 < M; i0 += tile_size) {
        for (int j0 = 0; j0 < N; j0 += tile_size) {
            for (int k0 = 0; k0 < K; k0 += tile_size) {
                // Process tile
                int i_end = std::min(i0 + tile_size, M);
                int j_end = std::min(j0 + tile_size, N);
                int k_end = std::min(k0 + tile_size, K);
                
                for (int i = i0; i < i_end; i++) {
                    for (int j = j0; j < j_end; j++) {
                        float sum = C[i * N + j];
                        for (int k = k0; k < k_end; k++) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

/**
 * Vectorized Matrix Multiplication (SIMD-friendly)
 * Tests vector instruction generation in backend
 */
void matmul_vectorized(const float* __restrict A, const float* __restrict B, 
                       float* __restrict C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * Initialize matrix with random values
 */
void init_matrix(float* M, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        M[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

/**
 * Compare two matrices for correctness
 */
bool compare_matrices(const float* A, const float* B, int size, float epsilon = EPSILON) {
    for (int i = 0; i < size; i++) {
        if (std::fabs(A[i] - B[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": " << A[i] << " vs " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * Print matrix (for debugging)
 */
void print_matrix(const float* M, int rows, int cols, const char* name) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < std::min(rows, 8); i++) {
        for (int j = 0; j < std::min(cols, 8); j++) {
            printf("%8.4f ", M[i * cols + j]);
        }
        if (cols > 8) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > 8) std::cout << "..." << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    int M = MATRIX_SIZE;
    int N = MATRIX_SIZE;
    int K = MATRIX_SIZE;
    
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    
    std::cout << "=========================================" << std::endl;
    std::cout << "MatMul Correctness Test for VEGA" << std::endl;
    std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Allocate matrices
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_naive(M * N);
    std::vector<float> C_tiled(M * N);
    std::vector<float> C_vectorized(M * N);
    
    // Initialize
    srand(42);  // Fixed seed for reproducibility
    init_matrix(A.data(), M, K);
    init_matrix(B.data(), K, N);
    
    std::cout << "\nRunning naive matmul..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    matmul_naive(A.data(), B.data(), C_naive.data(), M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Naive time: " << naive_time << " us" << std::endl;
    
    std::cout << "\nRunning tiled matmul..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    matmul_tiled(A.data(), B.data(), C_tiled.data(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    auto tiled_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Tiled time: " << tiled_time << " us" << std::endl;
    
    std::cout << "\nRunning vectorized matmul..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    matmul_vectorized(A.data(), B.data(), C_vectorized.data(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    auto vec_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Vectorized time: " << vec_time << " us" << std::endl;
    
    // Verify correctness
    std::cout << "\n=========================================" << std::endl;
    std::cout << "Correctness Verification" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    bool tiled_correct = compare_matrices(C_naive.data(), C_tiled.data(), M * N);
    std::cout << "Tiled vs Naive: " << (tiled_correct ? "PASS" : "FAIL") << std::endl;
    
    bool vec_correct = compare_matrices(C_naive.data(), C_vectorized.data(), M * N);
    std::cout << "Vectorized vs Naive: " << (vec_correct ? "PASS" : "FAIL") << std::endl;
    
    // Summary
    std::cout << "\n=========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "All tests " << ((tiled_correct && vec_correct) ? "PASSED" : "FAILED") << std::endl;
    
    return (tiled_correct && vec_correct) ? 0 : 1;
}
