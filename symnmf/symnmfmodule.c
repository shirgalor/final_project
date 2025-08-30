#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "symnmf.h"

// Convert numpy array to C double array
double** numpy_to_c_array(PyArrayObject* arr, int* rows, int* cols) {
    *rows = PyArray_DIM(arr, 0);
    *cols = PyArray_DIM(arr, 1);
    double** c_arr = alloc_matrix(*rows, *cols);
    if (!c_arr) return NULL;
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            c_arr[i][j] = *(double*)PyArray_GETPTR2(arr, i, j);
        }
    }
    return c_arr;
}

// Convert C double array to numpy array
PyObject* c_array_to_numpy(double** arr, int rows, int cols) {
    npy_intp dims[2] = {rows, cols};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) return NULL;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            *(double*)PyArray_GETPTR2((PyArrayObject*)result, i, j) = arr[i][j];
        }
    }
    return result;
}

static PyObject* py_symnmf(PyObject *self, PyObject *args) {
    PyArrayObject *X_array;
    int k;
    
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &X_array, &k)) {
        return NULL;
    }
    
    // Ensure array is contiguous and of correct type
    X_array = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)X_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (X_array == NULL) return NULL;
    
    int n, dim;
    double **points = numpy_to_c_array(X_array, &n, &dim);
    if (!points) {
        Py_DECREF(X_array);
        return NULL;
    }
    
    // Build similarity matrix
    double **A = alloc_matrix(n, n);
    if (!A) {
        free_matrix(points, n);
        Py_DECREF(X_array);
        return NULL;
    }
    build_similarity_matrix(points, n, dim, A);
    
    // Build degree matrix
    double **D = alloc_matrix(n, n);
    build_degree_matrix(A, n, D);
    
    // Build normalized similarity matrix
    double **W = alloc_matrix(n, n);
    build_normalized_similarity_matrix(A, D, n, W);
    
    // Calculate mean for initialization
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m += W[i][j];
        }
    }
    m /= (n * n);
    
    // Initialize H
    double **H = alloc_matrix(n, k);
    initialize_H(n, k, m, H);
    
    // Run update iterations
    for (int iter = 0; iter < 500; iter++) {
        update_H(H, W, n, k, 0.5);
    }
    
    // Convert result to numpy array
    PyObject *result = c_array_to_numpy(H, n, k);
    
    // Clean up
    free_matrix(points, n);
    free_matrix(A, n);
    free_matrix(D, n);
    free_matrix(W, n);
    free_matrix(H, n);
    Py_DECREF(X_array);
    
    return result;
}

static PyObject* py_sym(PyObject *self, PyObject *args) {
    PyArrayObject *X_array;
    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X_array)) {
        return NULL;
    }
    
    X_array = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)X_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (X_array == NULL) return NULL;
    
    int n, dim;
    double **points = numpy_to_c_array(X_array, &n, &dim);
    
    double **A = alloc_matrix(n, n);
    build_similarity_matrix(points, n, dim, A);
    
    PyObject *result = c_array_to_numpy(A, n, n);
    
    free_matrix(points, n);
    free_matrix(A, n);
    Py_DECREF(X_array);
    
    return result;
}

static PyObject* py_ddg(PyObject *self, PyObject *args) {
    PyArrayObject *X_array;
    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X_array)) {
        return NULL;
    }
    
    X_array = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)X_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (X_array == NULL) return NULL;
    
    int n, dim;
    double **points = numpy_to_c_array(X_array, &n, &dim);
    
    double **A = alloc_matrix(n, n);
    build_similarity_matrix(points, n, dim, A);
    
    double **D = alloc_matrix(n, n);
    build_degree_matrix(A, n, D);
    
    PyObject *result = c_array_to_numpy(D, n, n);
    
    free_matrix(points, n);
    free_matrix(A, n);
    free_matrix(D, n);
    Py_DECREF(X_array);
    
    return result;
}

static PyObject* py_norm(PyObject *self, PyObject *args) {
    PyArrayObject *X_array;
    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X_array)) {
        return NULL;
    }
    
    X_array = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)X_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (X_array == NULL) return NULL;
    
    int n, dim;
    double **points = numpy_to_c_array(X_array, &n, &dim);
    
    double **A = alloc_matrix(n, n);
    build_similarity_matrix(points, n, dim, A);
    
    double **D = alloc_matrix(n, n);
    build_degree_matrix(A, n, D);
    
    double **W = alloc_matrix(n, n);
    build_normalized_similarity_matrix(A, D, n, W);
    
    PyObject *result = c_array_to_numpy(W, n, n);
    
    free_matrix(points, n);
    free_matrix(A, n);
    free_matrix(D, n);
    free_matrix(W, n);
    Py_DECREF(X_array);
    
    return result;
}

// Method definitions
static PyMethodDef SymNMFMethods[] = {
    {"symnmf", py_symnmf, METH_VARARGS, "Perform symnmf and return H."},
    {"sym", py_sym, METH_VARARGS, "Calculate similarity matrix."},
    {"ddg", py_ddg, METH_VARARGS, "Calculate diagonal degree matrix."},
    {"norm", py_norm, METH_VARARGS, "Calculate normalized similarity matrix."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",   // name of module
    NULL,        // module documentation
    -1,          // size of per-interpreter state of the module
    SymNMFMethods
};

PyMODINIT_FUNC PyInit_symnmf(void) {
    import_array();  // Initialize NumPy C API
    return PyModule_Create(&symnmfmodule);
}
