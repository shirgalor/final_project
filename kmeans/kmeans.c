#define _GNU_SOURCE
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define EPSILON 0.001
#define LINE_CHUNK 128

typedef struct value {
    double* x;
    struct value* next;
} Value;

typedef struct value_node {
    Value* value;
    struct value_node* next;
} ValueNode;

typedef struct centroid {
    ValueNode* points;  
    double* x;           
} Centroid;

void free_old_centroids(double** old_centroids, int k) {
    int i;
    for (i = 0; i < k; i++) {
        free(old_centroids[i]);
    }
    free(old_centroids);
}

void free_centroids(Centroid* centroids, int k) {
    int i;
    ValueNode* curr;
    ValueNode* tmp;

    for (i = 0; i < k; i++) {
        free(centroids[i].x); /* free the centroid vector */

        /* free the list of ValueNodes */
        curr = centroids[i].points; 
        while (curr) {
            tmp = curr;
            curr = curr->next;
            free(tmp); /* do NOT free tmp->value – it's owned by the data list */
        }
    }
    free(centroids);
}

void free_data(Value* head) {
    while (head) {
        Value* tmp = head;
        head = head->next;
        free(tmp->x);   /* free the double array */
        free(tmp);      /* free the node itself */
    }
}

void clear_centroid_assignments(Centroid* centroids, int k) {
    int i;
    for (i = 0; i < k; i++) {
        ValueNode* curr = centroids[i].points;
        while (curr) {
            ValueNode* tmp = curr;
            curr = curr->next;
            free(tmp);
        }
        centroids[i].points = NULL;
    }
}

double euclidean(double* a, double* b, int dim) {
    double sum = 0.0;
    int i;
    for (i = 0; i < dim; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void assign_clusters(Value* data, Centroid* centroids, int k, int dim) {
    Value* current;
    int i, best;
    double min_dist, dist;
    ValueNode* node;

    clear_centroid_assignments(centroids, k);

    for (current = data; current != NULL; current = current->next) {
        best = 0;
        min_dist = euclidean(current->x, centroids[0].x, dim);
        for (i = 1; i < k; i++) {
            dist = euclidean(current->x, centroids[i].x, dim);
            if (dist < min_dist) {
                min_dist = dist;
                best = i;
            }
        }

        node = malloc(sizeof(ValueNode));
        node->value = current;
        node->next = centroids[best].points;
        centroids[best].points = node;
    }
}

void update_centroids(Centroid* centroids, int k, int dim) {
    int i, j, count;
    double* sum;
    ValueNode* node;

    for (i = 0; i < k; i++) {
        sum = calloc(dim, sizeof(double));
        count = 0;

        for (node = centroids[i].points; node != NULL; node = node->next) {
            for (j = 0; j < dim; j++) {
                sum[j] += node->value->x[j];
            }
            count++;
        }

        if (count > 0) {
            for (j = 0; j < dim; j++) {
                centroids[i].x[j] = sum[j] / count;
            }
        }

        free(sum);
    }
}

void kmeans(int k, int max_iter, Value* data, Centroid* centroids, int dim, double** old_centroids, double epsilon) {
    int iter = 0;
    int i, j = 0;
    double max_delta;

    do {
        for (i = 0; i < k; i++) {
            for (j = 0; j < dim; j++) {
                old_centroids[i][j] = centroids[i].x[j];
            }
        }

        assign_clusters(data, centroids, k, dim);
        update_centroids(centroids, k, dim);

        /* Check convergence */
        max_delta = 0.0;
        for (i = 0; i < k; i++) {
            double delta = 0.0;
            for (j = 0; j < dim; j++) {
                double diff = centroids[i].x[j] - old_centroids[i][j];
                delta += diff * diff;
            }
            delta = sqrt(delta);
            if (delta > max_delta) max_delta = delta;
        }

        iter++;
    } while (max_delta > epsilon && iter <= max_iter);
}