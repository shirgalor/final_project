#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"

// 1.1
//  Build the similarity matrix A from a list of points
// points: array of pointers to double arrays, n: number of points, dim: dimension of each point
void build_similarity_matrix(double **points, int n, int dim, double **A) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                double dist2 = 0.0;
                for (int k = 0; k < dim; ++k) {
                    double diff = points[i][k] - points[j][k];
                    dist2 += diff * diff;
                }
                A[i][j] = exp(-dist2 / 2.0);
            } else {
                A[i][j] = 0.0;
            }
        }
    }
}

// 1.2
//  Build the diagonal degree matrix D from similarity matrix A
void build_degree_matrix(double **A, int n, double **D) {
    for (int i = 0; i < n; ++i) {
        double d_i = 0.0;
        for (int j = 0; j < n; ++j) {
            d_i += A[i][j];
        }
        for (int j = 0; j < n; ++j) {
            D[i][j] = (i == j) ? d_i : 0.0;
        }
    }
}

// 1.3
//  Compute the normalized similarity matrix W = D^(-1/2) * A * D^(-1/2)
void build_normalized_similarity_matrix(double **A, double **D, int n, double **W) {
    double *D_inv_sqrt = (double *)malloc(n * sizeof(double));
    if (!D_inv_sqrt) return; // Error condition - caller should handle
    for (int i = 0; i < n; ++i) {
        D_inv_sqrt[i] = (D[i][i] > 0) ? 1.0 / sqrt(D[i][i]) : 0.0;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            W[i][j] = D_inv_sqrt[i] * A[i][j] * D_inv_sqrt[j];
        }
    }
    free(D_inv_sqrt);
}

// 1.4.1
//  Initialize H for SymNMF
void initialize_H(int n, int k, double m, double **H) {
    double upper = 2.0 * sqrt(m / k);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            H[i][j] = ((double)rand() / RAND_MAX) * upper;
        }
    }
}

// 1.4.2 and 1.4.3
// One SymNMF-style multiplicative update step
void update_H(double **H, double **W, int n, int k, double beta) {
    double eps = 1e-12;
    double **WH = (double **)malloc(n * sizeof(double *));
    double **HHTH = (double **)malloc(n * sizeof(double *));
    if (!WH || !HHTH) {
        if (WH) free(WH);
        if (HHTH) free(HHTH);
        return; // Error condition - caller should handle
    }
    for (int i = 0; i < n; ++i) {
        WH[i] = (double *)calloc(k, sizeof(double));
        HHTH[i] = (double *)calloc(k, sizeof(double));
        if (!WH[i] || !HHTH[i]) {
            // Free previously allocated memory
            for (int j = 0; j <= i; ++j) {
                if (WH[j]) free(WH[j]);
                if (HHTH[j]) free(HHTH[j]);
            }
            free(WH);
            free(HHTH);
            return; // Error condition - caller should handle
        }
    }
    // WH = W @ H
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            for (int l = 0; l < n; ++l) {
                WH[i][j] += W[i][l] * H[l][j];
            }
        }
    }
    // HHTH = H @ (H^T @ H)
    double **HTH = (double **)malloc(k * sizeof(double *));
    if (!HTH) {
        // Free previously allocated memory
        for (int i = 0; i < n; ++i) {
            free(WH[i]);
            free(HHTH[i]);
        }
        free(WH);
        free(HHTH);
        return; // Error condition - caller should handle
    }
    for (int i = 0; i < k; ++i) {
        HTH[i] = (double *)calloc(k, sizeof(double));
        if (!HTH[i]) {
            // Free previously allocated memory
            for (int j = 0; j < i; ++j) {
                free(HTH[j]);
            }
            free(HTH);
            for (int j = 0; j < n; ++j) {
                free(WH[j]);
                free(HHTH[j]);
            }
            free(WH);
            free(HHTH);
            return; // Error condition - caller should handle
        }
    }
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            for (int l = 0; l < n; ++l) {
                HTH[i][j] += H[l][i] * H[l][j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            for (int l = 0; l < k; ++l) {
                HHTH[i][j] += H[i][l] * HTH[l][j];
            }
        }
    }
    // Update H
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            double ratio = WH[i][j] / (HHTH[i][j] + eps);
            H[i][j] *= (1.0 - beta) + beta * ratio;
        }
    }
    // Free memory
    for (int i = 0; i < n; ++i) {
        free(WH[i]);
        free(HHTH[i]);
    }
    free(WH);
    free(HHTH);
    for (int i = 0; i < k; ++i) {
        free(HTH[i]);
    }
    free(HTH);
}

// Print matrix
void print_matrix(double **mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.4f%s", mat[i][j], (j == cols - 1) ? "\n" : ",");
        }
    }
}

// Derive hard clustering from H matrix
// For each row (element), find the column (cluster) with highest association score
int* hard_clustering(double **H, int n, int k) {
    int *clusters = (int *)malloc(n * sizeof(int));
    if (!clusters) return NULL;
    for (int i = 0; i < n; ++i) {
        int max_cluster = 0;
        double max_score = H[i][0];
        for (int j = 1; j < k; ++j) {
            if (H[i][j] > max_score) {
                max_score = H[i][j];
                max_cluster = j;
            }
        }
        clusters[i] = max_cluster;
    }
    return clusters;
}

// Helper to allocate a 2D array
double **alloc_matrix(int rows, int cols) {
    double **mat = (double **)malloc(rows * sizeof(double *));
    if (!mat) return NULL;
    for (int i = 0; i < rows; ++i) {
        mat[i] = (double *)malloc(cols * sizeof(double));
        if (!mat[i]) {
            // Free previously allocated memory
            for (int j = 0; j < i; ++j) {
                free(mat[j]);
            }
            free(mat);
            return NULL;
        }
    }
    return mat;
}

// Helper to free a 2D array
void free_matrix(double **mat, int rows) {
    for (int i = 0; i < rows; ++i) {
        free(mat[i]);
    }
    free(mat);
}

// Read points from a file
int read_points(const char *filename, double ***points, int *n, int *dim) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return 0;
    char line[4096];
    int count = 0;
    int d = 0;
    double **pts = NULL;
    int first_line = 1;
    
    while (fgets(line, sizeof(line), fp)) {
        char *p = line;
        int col = 0;
        double vals[256];
        int is_numeric_line = 1;
        
        // Check if this is the first line and if it contains non-numeric data (header)
        if (first_line) {
            char *temp_p = line;
            while (*temp_p && *temp_p != '\n') {
                if (isalpha(*temp_p)) {
                    is_numeric_line = 0;
                    break;
                }
                temp_p++;
            }
            first_line = 0;
            if (!is_numeric_line) {
                continue; // Skip header line
            }
        }
        
        while (*p) {
            while (*p == ' ' || *p == ',') ++p;
            if (*p == '\0' || *p == '\n') break;
            vals[col++] = strtod(p, &p);
        }
        if (col > 0) {
            double **temp_pts = (double **)realloc(pts, (count + 1) * sizeof(double *));
            if (!temp_pts) {
                // Free previously allocated memory
                for (int i = 0; i < count; ++i) {
                    free(pts[i]);
                }
                free(pts);
                fclose(fp);
                return 0;
            }
            pts = temp_pts;
            pts[count] = (double *)malloc(col * sizeof(double));
            if (!pts[count]) {
                // Free previously allocated memory
                for (int i = 0; i < count; ++i) {
                    free(pts[i]);
                }
                free(pts);
                fclose(fp);
                return 0;
            }
            for (int i = 0; i < col; ++i) pts[count][i] = vals[i];
            d = col;
            count++;
        }
    }
    fclose(fp);
    *points = pts;
    *n = count;
    *dim = d;
    return 1;
}
