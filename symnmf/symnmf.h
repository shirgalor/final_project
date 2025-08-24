/**
 * @file symnmf.h
 * @brief Header file for Symmetric Non-negative Matrix Factorization C library
 * 
 * Declares the main interface function for SymNMF multiplicative updates.
 * This library provides high-performance C implementations that can be
 * called from Python via the symnmfmodule.c extension.
 */

#ifndef SYMNMF_H
#define SYMNMF_H

#include <stddef.h>

/**
 * @brief Perform SymNMF multiplicative updates on factor matrix H
 * 
 * Main entry point for C-accelerated SymNMF computation. Updates H in-place
 * to minimize ||W - HH^T||_F^2 using multiplicative update rules.
 * 
 * @param H Factor matrix (n x k), modified in-place during optimization
 * @param W Similarity matrix (n x n), remains constant
 * @param n Number of data points (rows/columns of W, rows of H)  
 * @param k Number of factors/clusters (columns of H)
 * @param max_iter Maximum number of multiplicative update iterations
 * @param tol Relative convergence tolerance for early stopping
 * @param verbose Print iteration progress if non-zero
 * @return 0 on success, non-zero error code on failure
 */
int symnmf_update(double* H, const double* W, int n, int k, int max_iter, double tol, int verbose);

#endif
