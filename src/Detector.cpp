#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include "Detector.hpp"

const char *planFile = "tensornet.plan";

static DetectNet* net = nullptr;
//const int batchsize = 1;



extern "C" PyObject* netInit(PyObject* self, PyObject *args) {
    net = new DetectNet();   
    net->loadNet(planFile);
    return Py_BuildValue("f", 1.);
}

extern "C" PyObject* infer(PyObject* self, PyObject *args) {
    PyArrayObject *in_array;
    static int ret_tmp[12];
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)) {
        printf("errror in parse input args\n");
        return NULL;
    }
    int sz = PyArray_SIZE(in_array);
    float* data = (float*)PyArray_DATA(in_array);

    int batchsize = PyArray_NDIM(in_array) == 4 ? in_array->dimensions[0] : 1;
    float* result = (float*) malloc(sizeof(float) * batchsize * 4);
    net->inference(data, batchsize, result, sz);
    npy_intp output_dims[2] = {batchsize, 4};
    for (int i = 0; i < batchsize; i++) {
        int offset = i*4;
        ret_tmp[offset] = (result[offset] - result[offset + 2]/2)*640 + 0.5;
        ret_tmp[offset+1] = (result[offset] + result[offset + 2]/2)*640 + 0.5;
        ret_tmp[offset+2] = (result[offset+1] - result[offset + 3]/2)*360 + 0.5;
        ret_tmp[offset+3] = (result[offset+1] + result[offset + 3]/2)*360 + 0.5;
    }
    PyObject *ret = PyArray_SimpleNew(2, output_dims, NPY_INT);
    memcpy(PyArray_DATA(ret), ret_tmp, 4*batchsize*sizeof(int));
    //PyObject *ret = Py_BuildValue("ffff", result[0], result[1], result[2], result[3]);
    free(result);
    return ret;
}



void netDestroy() {
    delete net;
    net = nullptr;
}

static PyMethodDef methods[] = {
    { "netInit", (PyCFunction)netInit, METH_VARARGS, "net init"},
    {"infer", (PyCFunction)infer, METH_VARARGS, "net inference"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initmypack(void) {
    (void) Py_InitModule("mypack", methods);
    import_array();
    Py_Initialize();
}
