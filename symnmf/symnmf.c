#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

int main(int argc, char *argv[]) {
	if (argc != 3) {
		printf("An Error Has Occurred\n");
		return 1;
	}
	char *goal = argv[1];
	char *file_name = argv[2];

	double **points = NULL;
	int n = 0, dim = 0;
	if (!read_points(file_name, &points, &n, &dim)) {
		printf("An Error Has Occurred\n");
		return 1;
	}

	if (strcmp(goal, "sym") == 0) {
		double **A = alloc_matrix(n, n);
		if (!A) {
			printf("An Error Has Occurred\n");
			for (int i = 0; i < n; ++i) free(points[i]);
			free(points);
			return 1;
		}
		build_similarity_matrix(points, n, dim, A);
		print_matrix(A, n, n);
		free_matrix(A, n);
	} else if (strcmp(goal, "ddg") == 0) {
		double **A = alloc_matrix(n, n);
		if (!A) {
			printf("An Error Has Occurred\n");
			for (int i = 0; i < n; ++i) free(points[i]);
			free(points);
			return 1;
		}
		build_similarity_matrix(points, n, dim, A);
		double **D = alloc_matrix(n, n);
		if (!D) {
			printf("An Error Has Occurred\n");
			free_matrix(A, n);
			for (int i = 0; i < n; ++i) free(points[i]);
			free(points);
			return 1;
		}
		build_degree_matrix(A, n, D);
		print_matrix(D, n, n);
		free_matrix(A, n);
		free_matrix(D, n);
	} else if (strcmp(goal, "norm") == 0) {
		double **A = alloc_matrix(n, n);
		if (!A) {
			printf("An Error Has Occurred\n");
			for (int i = 0; i < n; ++i) free(points[i]);
			free(points);
			return 1;
		}
		build_similarity_matrix(points, n, dim, A);
		double **D = alloc_matrix(n, n);
		if (!D) {
			printf("An Error Has Occurred\n");
			free_matrix(A, n);
			for (int i = 0; i < n; ++i) free(points[i]);
			free(points);
			return 1;
		}
		build_degree_matrix(A, n, D);
		double **W = alloc_matrix(n, n);
		if (!W) {
			printf("An Error Has Occurred\n");
			free_matrix(A, n);
			free_matrix(D, n);
			for (int i = 0; i < n; ++i) free(points[i]);
			free(points);
			return 1;
		}
		build_normalized_similarity_matrix(A, D, n, W);
		print_matrix(W, n, n);
		free_matrix(A, n);
		free_matrix(D, n);
		free_matrix(W, n);
	} else {
		printf("An Error Has Occurred\n");
		return 1;
	}
	for (int i = 0; i < n; ++i) free(points[i]);
	free(points);
	return 0;
}
