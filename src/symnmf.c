#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

/* Helper function to print error and cleanup */
static void error_exit(double **points, int n) {
    printf("An Error Has Occurred\n");
    cleanup_points(points, n);
    exit(1);
}

/* Process sym goal */
static void process_sym(double **points, int n, int dim) {
    double **A = alloc_matrix(n, n);
    if (!A) error_exit(points, n);
    
    build_similarity_matrix(points, n, dim, A);
    print_matrix(A, n, n);
    free_matrix(A);
}

/* Process ddg goal */
static void process_ddg(double **points, int n, int dim) {
    double **A = alloc_matrix(n, n);
    double **D;
    if (!A) error_exit(points, n);
    
    build_similarity_matrix(points, n, dim, A);
    
    D = alloc_matrix(n, n);
    if (!D) {
        free_matrix(A);
        error_exit(points, n);
    }
    
    build_degree_matrix(A, n, D);
    print_matrix(D, n, n);
    
    free_matrix(A);
    free_matrix(D);
}

/* Process norm goal */
static void process_norm(double **points, int n, int dim) {
    double **A = alloc_matrix(n, n);
    double **D;
    double **W;
    if (!A) error_exit(points, n);
    
    build_similarity_matrix(points, n, dim, A);
    
    D = alloc_matrix(n, n);
    if (!D) {
        free_matrix(A);
        error_exit(points, n);
    }
    
    build_degree_matrix(A, n, D);
    
    W = alloc_matrix(n, n);
    if (!W) {
        free_matrix(A);
        free_matrix(D);
        error_exit(points, n);
    }
    
    build_normalized_similarity_matrix(A, D, n, W);
    print_matrix(W, n, n);
    
    free_matrix(A);
    free_matrix(D);
    free_matrix(W);
}

int main(int argc, char *argv[]) {
    char *goal;
    char *file_name;
    double **points = NULL;
    int n = 0, dim = 0;
    
    if (argc != 3) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    
    goal = argv[1];
    file_name = argv[2];
    
    /* Read input points */
    if (!read_points(file_name, &points, &n, &dim)) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    
    /* Process based on goal */
    if (strcmp(goal, "sym") == 0) {
        process_sym(points, n, dim);
    } else if (strcmp(goal, "ddg") == 0) {
        process_ddg(points, n, dim);
    } else if (strcmp(goal, "norm") == 0) {
        process_norm(points, n, dim);
    } else {
        printf("An Error Has Occurred\n");
        cleanup_points(points, n);
        return 1;
    }
    
    cleanup_points(points, n);
    return 0;
}
