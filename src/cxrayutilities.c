
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#ifdef __OPENMP__
#include <omp.h>
#endif

static PyObject* block_average1d(PyObject *self, PyObject *args);

static PyMethodDef XRU_Methods[] = {
    {"block_average1d",  block_average1d, METH_VARARGS,
     "block average for one-dimensional numpy array"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initcxrayutilities(void)
{
    PyObject *m;

    m = Py_InitModule("cxrayutilities", XRU_Methods);
    if (m == NULL)
        return;

    import_array();
}

static PyObject* block_average1d(PyObject *self, PyObject *args) {
//int block_average1d(double *block_av, double *input, int Nav, int N) {
    /*    block average for one-dimensional double array
     *
     *    Parameters
     *    ----------
     *    block_av:     block averaged output array
     *                  size = ceil(N/Nav) (out)
     *    input:        input array of double (in)
     *    Nav:          number of double to average
     *    N:            total number of input values
     */

    int i,j,Nav,N;
    PyArrayObject *input=NULL, *outarr=NULL;
    double *cin,*cout;
    double buf;

    // Python argument conversion code
    if (!PyArg_ParseTuple(args, "O!i",&PyArray_Type, &input, &Nav)) return NULL;
    
    if (PyArray_NDIM(input) != 1 || PyArray_TYPE(input) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double");
        return NULL; }
    N = PyArray_DIMS(input)[0];
    cin = PyArray_DATA(input);

    // create output ndarray
    npy_intp *nout=NULL;
    *nout = ((int)ceil(N/(float)Nav));
    outarr = (PyArrayObject *) PyArray_SimpleNew(1, nout, NPY_DOUBLE);
    cout = (double *) PyArray_DATA(outarr);
    
    // c-code following is performing the block averaging
    for(i=0; i<N; i=i+Nav) {
        buf=0;
        //perform one block average (j-i serves as counter -> last bin is therefore correct)
        for(j=i; j<i+Nav && j<N; ++j) {
            buf += cin[j];
        }
        cout[i/Nav] = buf/(float)(j-i); //save average to output array
    }
     
    // return output array
    return PyArray_Return(outarr);
}
