#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "kmeans.c"

static PyObject* fit(PyObject* self, PyObject* args) {
    PyObject* data_list;
    PyObject* initial_centroids_list;
    int k, max_iter;
    int dim = 0;
    int N, i, j;
    double epsilon;
    Value* data = NULL;
    Centroid* centroids = NULL;
    double** old_centroids = NULL;
    PyObject* result = NULL;

    // Parse Python arguments: data (list of lists), initial_centroids (list of lists), k (int), max_iter (int), epsilon (double)
    if (!PyArg_ParseTuple(args, "OOiid", &data_list, &initial_centroids_list, &k, &max_iter, &epsilon)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments");
        return NULL;
    }

    // Convert Python list to linked list of Value structs
    N = PyList_Size(data_list);
    if (N == 0) {
        PyErr_SetString(PyExc_ValueError, "Data list is empty");
        return NULL;
    }

    for (i = 0; i < N; i++) {
        PyObject* point = PyList_GetItem(data_list, i);
        if (!PyList_Check(point)) {
            PyErr_SetString(PyExc_ValueError, "Data points must be lists");
            free_data(data);
            return NULL;
        }

        dim = PyList_Size(point);
        Value* val = malloc(sizeof(Value));
        if (!val) {
            PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
            free_data(data);
            return NULL;
        }

        val->x = malloc(dim * sizeof(double));
        if (!val->x) {
            PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
            free(val);
            free_data(data);
            return NULL;
        }

        for (j = 0; j < dim; j++) {
            PyObject* coord = PyList_GetItem(point, j);
            if (!PyFloat_Check(coord) && !PyLong_Check(coord)) {
                PyErr_SetString(PyExc_ValueError, "Coordinates must be numbers");
                free(val->x);
                free(val);
                free_data(data);
                return NULL;
            }
            val->x[j] = PyFloat_AsDouble(coord);
        }

        val->next = data;
        data = val;
    }

    // Allocate centroids and old_centroids
    centroids = malloc(k * sizeof(Centroid));
    if (!centroids) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
        free_data(data);
        return NULL;
    }
    for (i = 0; i < k; i++) {
        centroids[i].x = malloc(dim * sizeof(double));
        if (!centroids[i].x) {
            for (j = 0; j < i; j++) free(centroids[j].x);
            free(centroids);
            free_data(data);
            PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
            return NULL;
        }
        centroids[i].points = NULL;
    }

    old_centroids = malloc(k * sizeof(double*));
    for (i = 0; i < k; i++) {
        old_centroids[i] = malloc(dim * sizeof(double));
        if (!old_centroids[i]) {
            for (j = 0; j < i; j++) free(old_centroids[j]);
            free(old_centroids);
            free_centroids(centroids, k);
            free_data(data);
            PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
            return NULL;
        }
    }

    // Initialize centroids from initial_centroids_list
    if (PyList_Size(initial_centroids_list) != k) {
        PyErr_SetString(PyExc_ValueError, "Initial centroids list size must match k");
        free_centroids(centroids, k);
        free_data(data);
        free_old_centroids(old_centroids, k);
        return NULL;
    }

    for (i = 0; i < k; i++) {
        PyObject* centroid = PyList_GetItem(initial_centroids_list, i);
        if (!PyList_Check(centroid) || PyList_Size(centroid) != dim) {
            PyErr_SetString(PyExc_ValueError, "Each centroid must be a list of the same dimension as data points");
            free_centroids(centroids, k);
            free_data(data);
            free_old_centroids(old_centroids, k);
            return NULL;
        }

        for (j = 0; j < dim; j++) {
            PyObject* coord = PyList_GetItem(centroid, j);
            if (!PyFloat_Check(coord) && !PyLong_Check(coord)) {
                PyErr_SetString(PyExc_ValueError, "Centroid coordinates must be numbers");
                free_centroids(centroids, k);
                free_data(data);
                free_old_centroids(old_centroids, k);
                return NULL;
            }
            centroids[i].x[j] = PyFloat_AsDouble(coord);
        }
    }

    // Run k-means algorithm
    kmeans(k, max_iter, data, centroids, dim, old_centroids, epsilon);

    // Prepare result as a Python list
    result = PyList_New(k);
    for (i = 0; i < k; i++) {
        PyObject* centroid = PyList_New(dim);
        for (j = 0; j < dim; j++) {
            PyList_SetItem(centroid, j, PyFloat_FromDouble(centroids[i].x[j]));
        }
        PyList_SetItem(result, i, centroid);
    }

    // Cleanup
    free_data(data);
    free_centroids(centroids, k);
    free_old_centroids(old_centroids, k);

    return result;
}

static PyMethodDef KMeansMethods[] = {
    {"fit", fit, METH_VARARGS, "Run k-means clustering"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanspp",
    "Python interface for k-means clustering",
    -1,
    KMeansMethods
};

PyMODINIT_FUNC PyInit_mykmeanspp(void) {
    return PyModule_Create(&kmeansmodule);
}
