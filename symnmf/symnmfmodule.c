/**
 * @file symnmfmodule.c
 * @brief Python C extension for accelerated SymNMF computations
 * 
 * This module provides a Python interface to the C implementation of
 * Symmetric Non-negative Matrix Factorization multiplicative updates.
 * Uses NumPy C API for efficient array handling.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "symnmf.h"

/**
 * @brief Python wrapper for SymNMF multiplicative updates
 * 
 * Performs in-place multiplicative updates on factor matrix H to minimize
 * ||W - HH^T||_F^2 using efficient C implementation.
 * 
 * @param self Module object (unused)
 * @param args Python tuple: (H, W, max_iter, tol, verbose)
 *             H: Factor matrix (n x k), modified in-place
 *             W: Similarity matrix (n x n), read-only
 *             max_iter: Maximum iterations (int)
 *             tol: Convergence tolerance (double) 
 *             verbose: Print progress (int, 0 or 1)
 * @return Python tuple (H_updated, obj_history) or NULL on error
 */
static PyObject* update_wrapper(PyObject* self, PyObject* args) {
    PyObject *H_obj, *W_obj;
    int max_iter;
    double tol;
    int verbose;
    if (!PyArg_ParseTuple(args, "OOidi", &H_obj, &W_obj, &max_iter, &tol, &verbose)) {
        return NULL;
    }

    PyArrayObject *H_arr = (PyArrayObject*)PyArray_FROM_OTF(H_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    PyArrayObject *W_arr = (PyArrayObject*)PyArray_FROM_OTF(W_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!H_arr || !W_arr) {
        Py_XDECREF(H_arr);
        Py_XDECREF(W_arr);
        return NULL;
    }

    int n = (int)PyArray_DIM(H_arr, 0);
    int k = (int)PyArray_DIM(H_arr, 1);

    if (PyArray_NDIM(W_arr) != 2 || (int)PyArray_DIM(W_arr, 0) != n || (int)PyArray_DIM(W_arr, 1) != n) {
        PyErr_SetString(PyExc_ValueError, "Shapes must be H(n,k), W(n,n)");
        Py_DECREF(H_arr);
        Py_DECREF(W_arr);
        return NULL;
    }

    double* H = (double*)PyArray_DATA(H_arr);
    const double* W = (const double*)PyArray_DATA(W_arr);

    int rc = symnmf_update(H, W, n, k, max_iter, tol, verbose);
    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError, "symnmf_update failed");
        Py_DECREF(H_arr);
        Py_DECREF(W_arr);
        return NULL;
    }

    npy_intp shape[1] = {1};
    PyObject* obj_hist = PyArray_SimpleNew(1, shape, NPY_DOUBLE);
    if (obj_hist) {
        double* data = (double*)PyArray_DATA((PyArrayObject*)obj_hist);
        data[0] = 0.0;
    }

    PyObject* H_ret = PyArray_Return(H_arr);
    Py_DECREF(W_arr);

    return Py_BuildValue("NN", H_ret, obj_hist);
}

/**
 * @brief Method definitions for the SymNMF Python module
 * 
 * Defines the available functions that can be called from Python.
 * Currently provides only the 'update' function for SymNMF computation.
 */
static PyMethodDef SymNMFMethods[] = {
    {"update", update_wrapper, METH_VARARGS, "Run SymNMF multiplicative updates in C: (H, obj_hist) = update(H, W, max_iter, tol, verbose)"},
    {NULL, NULL, 0, NULL}
};

/**
 * @brief Module definition structure for Python C extension
 * 
 * Defines module metadata including name, documentation, size, and methods.
 * The module provides C-accelerated SymNMF functionality to Python.
 */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_csymnmf",
    "C-accelerated SymNMF updates",
    -1,
    SymNMFMethods
};

/**
 * @brief Module initialization function
 * 
 * Called when the module is imported in Python. Initializes NumPy C API
 * and creates the module object with all defined methods.
 * 
 * @return PyObject* Module object or NULL on failure
 */
PyMODINIT_FUNC PyInit__csymnmf(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
