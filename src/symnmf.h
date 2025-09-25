#ifndef SYMNMF_H
#define SYMNMF_H

/* Core SymNMF algorithm functions */
void build_similarity_matrix(double **points, int n, int dim, double **A);
void build_degree_matrix(double **A, int n, double **D);
void build_normalized_similarity_matrix(double **A, double **D, int n, double **W);
void initialize_H(int n, int k, double m, double **H);
void update_H(double **H, double **W, int n, int k, double beta);
void update_H_single_step(double **H, double **W, int n, int k, double beta);

/* Matrix operations and utilities */
void matrix_multiply(double **A, double **B, double **C, int n, int m, int p);
double** compute_HTH(double **H, int n, int k);
void apply_multiplicative_update(double **H, double **WH, double **HHTH, int n, int k, double beta);
double frobenius_norm_squared_diff(double **H_new, double **H_old, int n, int k);
void copy_matrix(double **H_src, double **H_dst, int n, int k);

/* Clustering and output functions */
int* hard_clustering(double **H, int n, int k);
void print_matrix(double **mat, int rows, int cols);

/* Memory management functions */
double **alloc_matrix(int rows, int cols);
void free_matrix(double **mat);

/* File I/O functions */
int read_points(const char *filename, double ***points, int *n, int *dim);

#endif
