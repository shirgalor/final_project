#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "symnmf.h"

// Convert numpy array to C double array
double** numpy_to_c_array(PyArrayObject* arr, int* rows, int* cols) {
    *rows = PyArray_DIM(arr, 0);
    *cols = PyArray_DIM(arr, 1);
    double** c_arr = alloc_matrix(*rows, *cols);
    if (!c_arr) return NULL;
    // Copy data from numpy array to C array
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
    // Copy data from C array to numpy array
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            *(double*)PyArray_GETPTR2((PyArrayObject*)result, i, j) = arr[i][j];
        }
    }
    return result;
}

// Helper function to build the full SymNMF pipeline (A, D, W matrices)
static double** build_normalized_matrix(double **points, int n, int dim, double ***A_out, double ***D_out) {
    // Build similarity matrix
    double **A = alloc_matrix(n, n);
    if (!A) return NULL;
    build_similarity_matrix(points, n, dim, A);
    
    // Build degree matrix
    double **D = alloc_matrix(n, n);
    if (!D) {
        free_matrix(A);
        return NULL;
    }
    build_degree_matrix(A, n, D);
    
    // Build normalized similarity matrix
    double **W = alloc_matrix(n, n);
    if (!W) {
        free_matrix(A);
        free_matrix(D);
        return NULL;
    }
    build_normalized_similarity_matrix(A, D, n, W);
    
    *A_out = A;
    *D_out = D;
    return W;
}

// Helper function to parse single array input (for sym, ddg, norm functions)
static double** parse_single_array_input(PyObject *args, PyArrayObject **X_array, int *n, int *dim) {
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, X_array)) {
        return NULL;
    }
    
    *X_array = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)*X_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (*X_array == NULL) return NULL;
        
    return numpy_to_c_array(*X_array, n, dim);
}

// Helper function to cleanup and return result for single matrix functions
static PyObject* cleanup_and_return_matrix(double **points, double **result_matrix, 
                                         PyArrayObject *X_array, int n, int result_n) {
    PyObject *result = c_array_to_numpy(result_matrix, result_n, result_n);
    
    free_matrix(points);
    free_matrix(result_matrix);
    Py_DECREF(X_array);
    
    return result;
}

/**
 * Python wrapper for SymNMF given initial H and similarity matrix W
 * Input: numpy arrays H (n x k), W (n x n)
 * Output: updated H (n x k)
 */
static PyObject* py_symnmf(PyObject *self, PyObject *args) {
    PyArrayObject *H_array, *W_array;

    // Parse arguments: two NumPy arrays
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &H_array,
                          &PyArray_Type, &W_array)) {
        return NULL;
    }
    // Ensure both are double (float64) and contiguous
    H_array = (PyArrayObject*) PyArray_FROM_OTF((PyObject*) H_array, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    W_array = (PyArrayObject*) PyArray_FROM_OTF((PyObject*) W_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!H_array || !W_array) {
        Py_XDECREF(H_array);
        Py_XDECREF(W_array);
        return NULL;
    }
    // Get dimensions
    int n = PyArray_DIM(H_array, 0);
    int k = PyArray_DIM(H_array, 1);

    // Convert to C double** pointers
    double **H = numpy_to_c_array(H_array, &n, &k);
    double **W = numpy_to_c_array(W_array, &n, &n);
    if (!H || !W) {
        Py_DECREF(H_array);
        Py_DECREF(W_array);
        return NULL;
    }

    // Run update loop (beta hardcoded here as 0.5)
    update_H(H, W, n, k, 0.5);

    // Convert result back to NumPy
    PyObject *result = c_array_to_numpy(H, n, k);

    // Free C matrices if numpy_to_c_matrix malloc'd
    free_matrix(H);
    free_matrix(W);

    Py_DECREF(H_array);
    Py_DECREF(W_array);

    return result;
}

/**
 * Python wrapper for similarity matrix computation
 * Input: numpy array of data points (n x dim)
 * Output: similarity matrix A (n x n)
 */
static PyObject* py_sym(PyObject *self, PyObject *args) {
    PyArrayObject *X_array;
    int n, dim;
    
    // Parse input array
    double **points = parse_single_array_input(args, &X_array, &n, &dim);
    if (!points) {
        if (X_array) Py_DECREF(X_array);
        return NULL;
    }
    
    // Build similarity matrix A
    double **A = alloc_matrix(n, n);
    if (!A) {
        free_matrix(points);
        Py_DECREF(X_array);
        return NULL;
    }
    build_similarity_matrix(points, n, dim, A);
    
    // Return result
    return cleanup_and_return_matrix(points, A, X_array, n, n);
}

/**
 * Python wrapper for diagonal degree matrix computation
 * Input: numpy array of data points (n x dim)
 * Output: diagonal degree matrix D (n x n)
 */
static PyObject* py_ddg(PyObject *self, PyObject *args) {
    PyArrayObject *X_array;
    int n, dim;
    
    // Parse input array
    double **points = parse_single_array_input(args, &X_array, &n, &dim);
    if (!points) {
        if (X_array) Py_DECREF(X_array);
        return NULL;
    }
    
    // Build similarity matrix A
    double **A = alloc_matrix(n, n);
    if (!A) {
        free_matrix(points);
        Py_DECREF(X_array);
        return NULL;
    }
    build_similarity_matrix(points, n, dim, A);
    
    // Build degree matrix D
    double **D = alloc_matrix(n, n);
    if (!D) {
        free_matrix(points);
        free_matrix(A);
        Py_DECREF(X_array);
        return NULL;
    }
    build_degree_matrix(A, n, D);
    
    // Cleanup A and return D
    free_matrix(A);
    return cleanup_and_return_matrix(points, D, X_array, n, n);
}

/**
 * Python wrapper for normalized similarity matrix computation
 * Input: numpy array of data points (n x dim)
 * Output: normalized similarity matrix W (n x n)
 */
static PyObject* py_norm(PyObject *self, PyObject *args) {
    PyArrayObject *X_array;
    int n, dim;
    
    // Parse input array
    double **points = parse_single_array_input(args, &X_array, &n, &dim);
    if (!points) {
        if (X_array) Py_DECREF(X_array);
        return NULL;
    }
    
    // Build A, D, W matrices using existing helper
    double **A, **D;
    double **W = build_normalized_matrix(points, n, dim, &A, &D);
    if (!W) {
        free_matrix(points);
        Py_DECREF(X_array);
        return NULL;
    }
    // Cleanup intermediate matrices and return W
    free_matrix(A);
    free_matrix(D);
    return cleanup_and_return_matrix(points, W, X_array, n, n);
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
    "symnmf_c",   // name of module
    NULL,        // module documentation
    -1,          // size of per-interpreter state of the module
    SymNMFMethods
};

PyMODINIT_FUNC PyInit_symnmf_c(void) {
    import_array();  // Initialize NumPy C API
    return PyModule_Create(&symnmfmodule);
}
