#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "symnmf.h"

/* Constants */
static const double DIVISION_EPS = 1e-12;  /* Small value to prevent division by zero */
static const double CONVERGENCE_EPS = 1e-4;  /* Convergence threshold for SymNMF */
static const int MAX_ITERATIONS = 300;  /* Maximum iterations for SymNMF */

/* 1.1 */
/* Build the similarity matrix A from a list of points */
/* points: array of pointers to double arrays, n: number of points, dim: dimension of each point */
void build_similarity_matrix(double **points, int n, int dim, double **A) {
    int i, j, k;
    /* iterate over all pairs of points */
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            if (i != j) {
                double dist = 0.0;
                /* Compute squared Euclidean distance */
                for (k = 0; k < dim; ++k) {
                    double diff = points[i][k] - points[j][k];
                    dist += diff * diff;
                }
                /* Gaussian similarity */
                A[i][j] = exp(-dist / 2.0);
            } else {
                A[i][j] = 0.0;
            }
        }
    }
}

/* 1.2 */
/* Build the diagonal degree matrix D from similarity matrix A */
void build_degree_matrix(double **A, int n, double **D) {
    int i, j;
    /* iterate over all nodes to compute degrees */
    for (i = 0; i < n; ++i) {
        double d_i = 0.0;
        /* Sum the i-th row of A to get the degree */
        for (j = 0; j < n; ++j) {
            d_i += A[i][j];
        }
        /* Fill diagonal entry and set off-diagonal to 0 */
        for (j = 0; j < n; ++j) {
            D[i][j] = (i == j) ? d_i : 0.0;
        }
    }
}

/* 1.3 */
/* Compute the normalized similarity matrix W = D^(-1/2) * A * D^(-1/2) */
void build_normalized_similarity_matrix(double **A, double **D, int n, double **W) {
    double *D_inv_sqrt;
    int i, j;
    
    D_inv_sqrt = (double *)malloc(n * sizeof(double));
    if (!D_inv_sqrt) return;
    
    /* Compute D^(-1/2) */
    for (i = 0; i < n; ++i) {
        D_inv_sqrt[i] = (D[i][i] > 0) ? 1.0 / sqrt(D[i][i]) : 0.0;
    } 
    /* Compute W = D^(-1/2) * A * D^(-1/2) */
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            W[i][j] = D_inv_sqrt[i] * A[i][j] * D_inv_sqrt[j];
        }
    }
    /* Free temporary array */
    free(D_inv_sqrt);
}

/* Helper function to compute matrix multiplication: C = A * B */
/* A is n x m, B is m x p, C is n x p */
void matrix_multiply(double **A, double **B, double **C, int n, int m, int p) {
    int i, j, l;
    /* iterate over all rows of A */
    for (i = 0; i < n; ++i) {
        for (j = 0; j < p; ++j) {
            C[i][j] = 0.0;
            /* sum over the shared dimension */
            for (l = 0; l < m; ++l) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

/* Helper function to compute H^T * H */
/* H is n x k, result HTH is k x k */
double** compute_HTH(double **H, int n, int k) {
    double **HTH = alloc_matrix(k, k);
    int i, j, l;
    if (!HTH) return NULL;

    /* iterate over all pairs of clusters */
    for (i = 0; i < k; ++i) {
        for (j = 0; j < k; ++j) {
            HTH[i][j] = 0.0;
            /* sum over all data points */
            for (l = 0; l < n; ++l) {
                HTH[i][j] += H[l][i] * H[l][j];
            }
        }
    }
    return HTH;
}

/* Helper function to apply multiplicative update rule */
void apply_multiplicative_update(double **H, double **WH, double **HHTH, int n, int k, double beta) {
    int i, j;
    /* iterate over all elements of H */
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            double ratio = WH[i][j] / (HHTH[i][j] + DIVISION_EPS);
            /* Update H with the multiplicative rule */
            H[i][j] *= (1.0 - beta) + beta * ratio;
        }
    }
}

/* Helper function to compute Frobenius norm squared of difference  */
/* between two matrices */
double frobenius_norm_squared_diff(double **H_new, double **H_old, int n, int k) {
    double norm_sq = 0.0;
    int i, j;
    /* iterate over all elements */
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            /* Compute the difference */
            double diff = H_new[i][j] - H_old[i][j];
            norm_sq += diff * diff;
        }
    }
    return norm_sq;
}

/* Helper function to copy matrix H_src to H_dst */
void copy_matrix(double **H_src, double **H_dst, int n, int k) {
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            H_dst[i][j] = H_src[i][j];
        }
    }
}

/* 1.4.2  */
/* One SymNMF-style multiplicative update step */
void update_H_single_step(double **H, double **W, int n, int k, double beta) {
    /* Allocate matrices for computations */
    double **WH = alloc_matrix(n, k);
    double **HTH = compute_HTH(H, n, k);
    double **HHTH = alloc_matrix(n, k);
    
    /* Check if any allocation failed */
    if (!WH || !HTH || !HHTH) {
        if (WH) free_matrix(WH);
        if (HTH) free_matrix(HTH);
        if (HHTH) free_matrix(HHTH);
        return; /* Error condition */
    }
    matrix_multiply(W, H, WH, n, n, k); /* Compute WH = W * H */
    matrix_multiply(H, HTH, HHTH, n, k, k); /* Compute HHTH = H * HTH */
    apply_multiplicative_update(H, WH, HHTH, n, k, beta); /* Apply multiplicative update rule */
    
    /* Free temporary matrices */
    free_matrix(WH);
    free_matrix(HTH);
    free_matrix(HHTH);
}

/* 1.4.3 */
/* Full SymNMF algorithm with convergence checking */
void update_H(double **H, double **W, int n, int k, double beta) {
    /* Allocate matrix to store previous H for convergence checking */
    double **H_prev = alloc_matrix(n, k);
    int iter;
    double norm_diff;
    if (!H_prev) return;
    
    for (iter = 0; iter < MAX_ITERATIONS; ++iter) {
        /* Save current H as previous */
        copy_matrix(H, H_prev, n, k);
        
        /* Perform one update step */
        update_H_single_step(H, W, n, k, beta);
        
        /* Check convergence: ||H^(t+1) - H^(t)||²_F < ε */
        norm_diff = frobenius_norm_squared_diff(H, H_prev, n, k);
        if (norm_diff < CONVERGENCE_EPS) {
            break; /* Converged */
        }
    }
    /* Free temporary matrix */
    free_matrix(H_prev);
}

/* Print matrix */
void print_matrix(double **mat, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            printf("%.4f%s", mat[i][j], (j == cols - 1) ? "\n" : ",");
        }
    }
}

/* Helper to allocate a 2D array */
double **alloc_matrix(int rows, int cols) {
    double **mat = malloc(rows * sizeof(double*));
    double *data = malloc(rows * cols * sizeof(double));
    int i;
    
    if (!mat) return NULL;

    if (!data) {
        free(mat);
        return NULL;
    }

    for (i = 0; i < rows; ++i) {
        mat[i] = data + i * cols;
    }

    return mat;
}

/* Helper to free a 2D contiguous array */
void free_matrix(double **mat) {
    free(mat[0]);
    free(mat);
}

/* Helper function to check if a line contains only numeric data */
static int is_numeric_line(const char *line) {
    const char *p = line;
    while (*p && *p != '\n') {
        if (isalpha(*p)) {
            return 0; /* Found alphabetic character */
        }
        p++;
    }
    return 1; /* Only numeric/punctuation characters */
}

/* Helper function to add a new point to the points array */
static int add_point(double ***pts, int *count, const double *vals, int dim) {
    /* Reallocate array for one more point */
    double **temp_pts = (double **)realloc(*pts, (*count + 1) * sizeof(double *));
    int i;
    if (!temp_pts) {
        return 0; /* Allocation failed */
    }
    *pts = temp_pts;
    
    /* Allocate memory for the new point */
    (*pts)[*count] = (double *)malloc(dim * sizeof(double));
    if (!(*pts)[*count]) {
        return 0; /* Allocation failed */
    }
    
    /* Copy values to the new point */
    for (i = 0; i < dim; ++i) {
        (*pts)[*count][i] = vals[i];
    }
    (*count)++;
    return 1; /* Success */
}

/* Helper function to cleanup allocated points on error */
void cleanup_points(double **pts, int count) {
    int i;
    if (pts) {
        for (i = 0; i < count; ++i) {
            free(pts[i]);
        }
        free(pts);
    }
}

/* Helper function to parse a line and extract numeric values dynamically */
static int parse_line_dynamic(char *line, double **vals, size_t *vals_capacity) {
    char *p = line;
    int col = 0;
    double *temp_vals;
    
    while (*p) {
        /* Skip whitespace and commas */
        while (*p == ' ' || *p == ',') ++p;
        if (*p == '\0' || *p == '\n') break;
        
        /* Resize vals array if needed (double the capacity) */
        if ((size_t)col >= *vals_capacity) {
            *vals_capacity *= 2;
            temp_vals = realloc(*vals, *vals_capacity * sizeof(double));
            if (!temp_vals) return -1;
            *vals = temp_vals;
        }
        
        /* Parse the number */
        (*vals)[col++] = strtod(p, (char**)&p);
    }
    return col;
}

/* Helper function to process a single line from the file */
static int process_line(char *line, int *first_line, double **vals, size_t *vals_capacity,
                       double ***pts, int *count) {
    int col;
    /* Skip header line if it's non-numeric */
    if (*first_line) {
        *first_line = 0;
        if (!is_numeric_line(line)) {
            return 0; /* Skip this line */
        }
    }
    
    /* Parse values from the line */
    col = parse_line_dynamic(line, vals, vals_capacity);
    if (col < 0) return -1;
    if (col == 0) return 0;  /* Empty line */
    
    /* Add the point to the array */
    if (!add_point(pts, count, *vals, col)) {
        return -1;
    }
    return col;
}

/* Read points from a file */
int read_points(const char *filename, double ***points, int *n, int *dim) {
    FILE *fp = fopen(filename, "r");
    size_t line_capacity = 0;
    char *line = NULL;
    size_t vals_capacity = 64;
    double *vals;
    int count = 0, d = 0, first_line = 1;
    double **pts = NULL;
    int line_len;
    
    if (!fp) return 0;
    
    /* Dynamic allocation - no fixed limits */
    vals = malloc(vals_capacity * sizeof(double));
    if (!vals) {
        fclose(fp);
        return 0;
    }
    
    /* Process each line */
    while ((line_len = getline(&line, &line_capacity, fp)) != -1) {
        int result = process_line(line, &first_line, &vals, &vals_capacity, &pts, &count);
        if (result < 0) {
            free(line);
            free(vals);
            cleanup_points(pts, count);
            fclose(fp);
            return 0;
        }
        if (result > 0) d = result; /* Update dimension */
    }
    /* Cleanup and return results */
    free(line);
    free(vals);
    fclose(fp);
    *points = pts;
    *n = count;
    *dim = d;
    return 1;
}
