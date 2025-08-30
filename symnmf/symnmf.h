#ifndef SYMNMF_H
#define SYMNMF_H

// Function declarations for symnmf_tools.c
void build_similarity_matrix(double **points, int n, int dim, double **A);
void build_degree_matrix(double **A, int n, double **D);
void build_normalized_similarity_matrix(double **A, double **D, int n, double **W);
void initialize_H(int n, int k, double m, double **H);
void update_H(double **H, double **W, int n, int k, double beta);
void print_matrix(double **mat, int rows, int cols);
int* hard_clustering(double **H, int n, int k);
double **alloc_matrix(int rows, int cols);
void free_matrix(double **mat, int rows);
int read_points(const char *filename, double ***points, int *n, int *dim);

#endif
