#include <Python.h>

int CFib(int n)
{
	if (n < 2)
		return n;
	else 
		return Cfib(n-1) + Cfib(n-2);
}

static PyObject* fib(PyObject* self, PyObject* args)
{
	int n;
	if(!PyArg_ParseTuple(args, "i", &n)) //parse arg for an integer and store in n
		return NULL;
	return Py_BuildValue("i", Cfib(n));// return an int with the result of cfib(n)
}

static PyObject* version(PyObject* self)
{
	return Py_BuildValue("s", "Version 1.0");
}

static PyMethodDef myMethods[] = {
	{"fib", fib, METH_VARARGS, "calculated the fib #s "},
	{"version", (PyCFunction)version, METH_NOARGS, "returns the version."},
	{NULL, NULL, 0, NULL} // python won't throw compiler errors without this!!
};

static struct PyModuleDef myModule = {
	PyModuleDef_HEAD_INIT,// method def head 
	"myModule",
	"Fibonacci Module", // documentation
	-1, //global state
	myMethods // methodDef
};

PyMODINIT_FUNC PyInit_myModule(void)
{
	return PyModule_Create(&myModule);
}

