
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
    PyObject *output=NULL, *input=NULL;
    PyArrayObject *inarr=NULL, *outarr=NULL;
    double *cin,*cout;
    double buf;

    // Python argument conversion code
    if (!PyArg_ParseTuple(args, "Oi", &input, &Nav)) return NULL;
    
    inarr = (PyArrayObject *) PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (inarr == NULL) return NULL;
    if (PyArray_NDIM(inarr) != 1) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional");
        return NULL; }
    N = PyArray_SHAPE(inarr)[0];
    
    cin = PyArray_DATA(inarr);
    
    // create output ndarray
    npy_intp *nout;
    *nout = ((int)ceil(N/(float)Nav));
    outarr = (PyArrayObject *) PyArray_SimpleNew(1, nout, NPY_DOUBLE);
    cout = (double *) PyArray_DATA(outarr);
    
    printf("%d %d\n",N,Nav);
    
    // c-code following is performing the block averaging
    for(i=0; i<N; i=i+Nav) {
        printf("i %d\n",i);
        buf=0;
        //perform one block average (j-i serves as counter -> last bin is therefore correct)
        for(j=i; j<i+Nav && j<N; ++j) {
            printf("%d %d\n",i,j);
            buf += cin[j];
        }
        cout[i/Nav] = buf/(float)(j-i); //save average to output array
        printf("cout %f\n",cout[i/Nav]);
    }
    
    //PyErr_SetString(PyExc_ValueError,"dummy error");
    //return NULL;
    
    Py_DECREF(inarr);
    Py_DECREF(outarr);
    // return output array
    return PyArray_Return(outarr);
}
