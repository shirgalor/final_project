/**
 * @file symnmf.c
 * @brief Core C implementation of Symmetric Non-negative Matrix Factorization
 * 
 * Provides efficient C implementations of SymNMF multiplicative updates
 * with optimized matrix operations and memory management.
 */

#include "symnmf.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/**
 * @brief Compute matrix multiplication C = A * B
 * 
 * Performs dense matrix multiplication using standard triple-loop algorithm.
 * Matrices are stored in row-major order.
 * 
 * @param A Left matrix (n x m)
 * @param B Right matrix (m x p) 
 * @param C Output matrix (n x p), overwritten
 * @param n Number of rows in A and C
 * @param m Number of columns in A, rows in B
 * @param p Number of columns in B and C
 */
static void matmul_nn(const double* A, const double* B, double* C, int n, int m, int p) {
    // Triple nested loop for standard matrix multiplication C = A * B
    for (int i = 0; i < n; ++i) {          // Iterate over rows of A and C
        for (int j = 0; j < p; ++j) {      // Iterate over columns of B and C
            double s = 0.0;               // Accumulator for dot product
            for (int t = 0; t < m; ++t) { // Inner product: A[i,:] · B[:,j]
                s += A[i*m + t] * B[t*p + j];  // Row-major indexing
            }
            C[i*p + j] = s;               // Store result
        }
    }
}

/**
 * @brief Compute Frobenius norm squared of reconstruction error ||W - HH^T||_F^2
 * 
 * Calculates the SymNMF objective function by computing HH^T and measuring
 * the squared Frobenius norm of the difference with the target matrix W.
 * 
 * @param W Target similarity matrix (n x n)
 * @param H Factor matrix (n x k)
 * @param n Number of data points
 * @param k Number of factors/clusters
 * @return Frobenius norm squared, or -1.0 on memory allocation failure
 */
static double frob_norm_sq_diff(const double* W, const double* H, int n, int k) {
    // Allocate temporary matrix for HH^T computation
    double* HHt = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!HHt) return -1.0;  // Memory allocation failed
    
    // Compute HH^T: (HH^T)[i,j] = sum_t H[i,t] * H[j,t]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int t = 0; t < k; ++t) {
                s += H[i*k + t] * H[j*k + t];  // Inner product of rows i and j
            }
            HHt[i*n + j] = s;
        }
    }
    
    // Compute ||W - HH^T||_F^2 = sum_{i,j} (W[i,j] - HH^T[i,j])^2
    double acc = 0.0;
    for (int i = 0; i < n*n; ++i) {
        double d = W[i] - HHt[i];   // Element-wise difference
        acc += d*d;                 // Square and accumulate
    }
    
    free(HHt);  // Clean up temporary memory
    return acc;
}

/**
 * @brief Perform SymNMF multiplicative updates to minimize ||W - HH^T||_F^2
 * 
 * Iteratively updates factor matrix H using multiplicative rule:
 * H[i,a] *= (WH)[i,a] / (HH^TH)[i,a]
 * 
 * Implements convergence checking and optional verbose progress reporting.
 * Uses numerical stabilization to prevent NaN/infinity values.
 * 
 * @param H Factor matrix (n x k), modified in-place
 * @param W Similarity matrix (n x n), read-only
 * @param n Number of data points
 * @param k Number of factors/clusters  
 * @param max_iter Maximum number of iterations
 * @param tol Relative convergence tolerance
 * @param verbose Print iteration progress if non-zero
 * @return 0 on success, 1 on memory allocation failure, 2 on objective computation failure
 */
int symnmf_update(double* H, const double* W, int n, int k, int max_iter, double tol, int verbose) {
    const double eps = 1e-12;
    double* WH = (double*)malloc((size_t)n * (size_t)k * sizeof(double));
    double* HtH = (double*)malloc((size_t)k * (size_t)k * sizeof(double));
    double* HHtH = (double*)malloc((size_t)n * (size_t)k * sizeof(double));
    if (!WH || !HtH || !HHtH) {
        free(WH); free(HtH); free(HHtH);
        return 1;
    }

    double obj_prev = frob_norm_sq_diff(W, H, n, k);
    if (obj_prev < 0.0) { free(WH); free(HtH); free(HHtH); return 2; }

    for (int it = 0; it < max_iter; ++it) {
        matmul_nn(W, H, WH, n, n, k);

        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < k; ++b) {
                double s = 0.0;
                for (int i = 0; i < n; ++i) {
                    s += H[i*k + a] * H[i*k + b];
                }
                HtH[a*k + b] = s;
            }
        }

        matmul_nn(H, HtH, HHtH, n, k, k);

        for (int i = 0; i < n; ++i) {
            for (int a = 0; a < k; ++a) {
                double denom = HHtH[i*k + a] + eps;
                double val = H[i*k + a] * (WH[i*k + a] / denom);
                if (isnan(val) || isinf(val) || val < 0.0) {
                    val = 0.0;
                }
                H[i*k + a] = val;
            }
        }

        double obj = frob_norm_sq_diff(W, H, n, k);
        if (verbose) {
            double rel = (obj_prev - obj) / (obj_prev > 1.0 ? obj_prev : 1.0);
            printf("[c ] iter=%03d, obj=%.6e, rel_change=%.3e\n", it+1, obj, rel);
        }
        if (fabs(obj_prev - obj) / (obj_prev > 1.0 ? obj_prev : 1.0) < tol) {
            break;
        }
        obj_prev = obj;
    }

    free(WH); free(HtH); free(HHtH);
    return 0;
}
