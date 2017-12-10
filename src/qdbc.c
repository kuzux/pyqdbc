/*
 * Copyright (c) 2017 Arda Cinar
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
**/

#include <Python.h>
/* Apparently datetimes in python need to be imported and initialized separately */
#include <datetime.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL qdbc
#include <numpy/arrayobject.h>

#define KXVER 3
#include <k.h>

/* we still need this outside debug mode because we need to use snprintf */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/* Can we definitely work on windows? 
 * TODO: fix the build script on windows
 */
#include <fcntl.h>
#include <errno.h>

#define BUFSIZE 255

#define DEBUG
#ifdef DEBUG
#define TRACE(fmt, ...) do{ fprintf(stderr, #fmt "\n", ## __VA_ARGS__); } while(0);
#else
#define TRACE
#endif

#ifdef _WIN32 /* Windows-related shit, basically */
  #define F_GETFL 3
  void __security_check_cookie(int i) {};
  int fcntl (intptr_t fd, int cmd, ...) { return (fd>=0)?1:-1; }
#endif

#define DATE_OFFSET 10957
#define DAYS_TO_SECS 86400
#define EPOCH_OFFSET_SECS 946677600

static PyObject* QdbcError;

/* TODO how to handle infinite or null values? */

/* 
 * J/K family of languages are known for their particular style of naming variables
 * and this style extends to their C sources as well (At least it does in J's case,
 * we can see its source. For K, it definitely applies to its C bindings, where most
 * types and functions' names are only a few characters long, and it's not obvious
 * from the name (at least not to me) what do they do.
 *
 * Little explanation of stuff we use:
 * the type K is a typedef to a pointer to a struct (that's why all those functions simply
 * take a K value, not a pointer to it), which is more-or-less a tagged union. One of the 
 * members, t, is the type of the value and it is the 'tag' in this 'tagged union'. We 
 * obviously math on its value
 * the function khp(char*, int) -> int is the function establishing connection to a running
 * KDB/Q instance. It needs to have its port set, either by \p command or by a command line
 * argument. The function returns us a raw file descriptor. On connection failure, it returns 
 * a negative number
 * the function k(int, char*) -> K is our main query function, it takes a file descriptor 
 * returned by khp and a query; and runs that query in the remote Q instance. It returns
 * NULL on connection failure and on compiler error; returns a valid K object whose type 
 * is -255. (The error message is contained in a member variable
 * kclose(int) -> void: the only function in use with a self-explanatory name 
 */

PyObject* get_atom(K k) {
    TRACE("reading atom");

    PyObject* tmp;

    switch(k->t) {
        case -1: return PyBool_FromLong(k->g);                      /* bool  */
        case -4: return PyInt_FromLong(k->g);                       /* byte  */ 
        case -5: return PyInt_FromLong(k->h);                       /* short */
        case -6: return PyLong_FromLong(k->i);                      /* int   */
        case -7: return PyLong_FromLong(k->j);                      /* long  */
        case -8: return PyFloat_FromDouble(k->e);                   /* real  */
        case -9: return PyFloat_FromDouble(k->f);                   /* float */
        case -10: return PyString_FromStringAndSize((char*)&k->i, 1); /* char */
        case -11: return PyString_FromString(k->s);                 /* symbol */
        case -12: 
            /* timestamp */
            {
                long micros = (k->j%1000000000)/1000;
                long secs = (k->j/1000000000);
                double us = (micros/1000000.0);

                tmp = PyFloat_FromDouble(EPOCH_OFFSET_SECS+secs+us);
                tmp = Py_BuildValue("(o)", tmp);
                return PyDateTime_FromTimestamp(tmp);
            }
        case -13:
            /* month */
            {
                int months = 1 + (k->i)%12;
                int years = 2000 + (k->i)/12;

                return PyDateTime_FromDateAndTime(years, months, 1, 0, 0, 0, 0);
            }
        case -14:
            /* date */
            tmp = Py_BuildValue("(i)", EPOCH_OFFSET_SECS+DAYS_TO_SECS*k->i);
            return PyDateTime_FromTimestamp(tmp);
        case -15:
            /* datetime */
            /* Q datetime objs are weird, in that they are floating point numbers
             * where the part after point shows the time.
             * It's advised to not use them in Q, and I'm not bothering
             * to do the conversion yet 
             */
            TRACE("%f", k->f);
            break;
        case -16:
            /* timespan */
            break;
        case -17:
            /* minute */
            break;
        case -18:
            /* second */
            break;
        case -19:
            /* time */
            break;
    }

    PyErr_SetString(QdbcError, "Atom type unknown");
    return NULL;
}

static PyObject* get_mixed_list(K k) {
    TRACE("getting mixed list");
    /* We need to convert this type of heterogeneus list 
     * to a classic python list, not a numpy array */
    return NULL;
}

static PyObject* get_list(K k) {
    /* function stub */
    TRACE("getting list");

    /* list of length n = dimensions are (n,) */
    size_t len = k->n;
    long num_dims = 1;

    int numpy_arr_type = 0;

    switch(k->t) {
        case 1: return PyBool_FromLong(k->g);                      /* bool  */
        case 4: return PyInt_FromLong(k->g);                       /* byte  */ 
        case 5: return PyInt_FromLong(k->h);                       /* short */
        case 6: return PyLong_FromLong(k->i);                      /* int   */
        case 7: return PyLong_FromLong(k->j);                      /* long  */
        case 8: return PyFloat_FromDouble(k->e);                   /* real  */
        case 9: return PyFloat_FromDouble(k->f);                   /* float */
        case 10: return PyString_FromStringAndSize((char*)&k->i, 1); /* char */
        case 11: return PyString_FromString(k->s);                 /* symbol */
        case 12: 
            /* timestamp */
            break;
    }

    return NULL;
}

static PyObject* get_dict(K k) {
    /* function stub */
    TRACE("getting dictionary");
    return NULL;
}

static PyObject* get_table(K k) {
    /* function stub */
    TRACE("getting table");
    return NULL;
}

static PyObject* qdbc_query(PyObject* self, PyObject* args) {
    char* host;
    int port;
    char* query;

    if(!PyArg_ParseTuple(args, "sis"/* atma oc */, &host, &port, &query)) {
        return NULL;
    }

    /* There used to be an empty check for the query, but apparently
     * Q just returns void on empty query and doesn't complain
     * so the check was kind of superfluous */

    int conn;
    conn = khp(host, port);
    TRACE("connection fd %d", conn);

    /* Check the connection */
    if(fcntl(conn, F_GETFL) < 0 || errno == EBADF) {
        PyErr_SetString(QdbcError, "Couldn't connect");
        return NULL;
    }

    K res;
    res = k(conn, query, NULL);

    if(!res) {
        PyErr_SetString(QdbcError, "Can't get query result");
        return NULL;
    }

    if(res->t == -128) {
        char buf[BUFSIZE];
        snprintf(buf, BUFSIZE, "Q error: %s", res->s);
        PyErr_SetString(QdbcError, buf);

        TRACE("closing connection");
        kclose(conn);
        
        return NULL;
    }

    if (res->t == 101) {
        TRACE("getting void");
        /* we have a void object */
        Py_INCREF(Py_None);
        TRACE("closing connection");
        kclose(conn);
        return Py_None;
    }

    PyObject* retVal = NULL;

    TRACE("Object type %d\n", res->t);

    if(res->t < 0) {
        retVal = get_atom(res);
    } else if(res->t==0) {
        retVal = get_mixed_list(res);
    } else if(res->t == 98) {
        retVal = get_table(res);
    } else if(res->t == 99) {
        retVal = get_dict(res);
    } else {
        retVal = get_list(res);
    }

    if(!retVal) {
        PyErr_SetString(QdbcError, "Deserializing this type is not yet implemented");
        return NULL;
    }

    TRACE("closing connection");
    kclose(conn);

    return retVal;
}

static PyObject* qdbc_testnumpy(PyObject* self, PyObject* args) {
    double* vals = (double*)malloc(2*sizeof(double));
    vals[0] = 3.5;
    vals[1] = 7.8;

    long dims[1];
    dims[0] = 2;
    int numDims = 1;

    PyObject* res = PyArray_SimpleNewFromData(numDims, dims, NPY_DOUBLE, vals);
    PyArray_ENABLEFLAGS((PyArrayObject*)res, NPY_ARRAY_OWNDATA);
    return res;
}

static PyObject* qdbc_ping(PyObject* self, PyObject* args) {
    char* host;
    int port;
    if(!PyArg_ParseTuple(args, "si", &host, &port)) {
        return NULL;
    }

    int conn = khp(host, port);
    TRACE("conn fd %d", conn);

    if(fcntl(conn, F_GETFL) < 0 || errno == EBADF) {
        Py_INCREF(Py_False);
        return Py_False;
    }

    K reply = k(conn, "::");
    TRACE("ping reply type %d", reply->t);

    TRACE("closing connection");
    kclose(conn);

    if(reply->t == 101) {
        Py_INCREF(Py_True);
        return Py_True;
    } else {
        Py_INCREF(Py_False);
        return Py_False;
    }
}

/* TODO: Write qopen/qclose type functions to keep a single session open */

static PyMethodDef QdbcMethods[] = {
    {"query", qdbc_query, METH_VARARGS,
        "Run a KDB query"},
    {"ping", qdbc_ping, METH_VARARGS,
        "Checks if a Q instance is healthy"},
    {"testnumpy", qdbc_testnumpy, METH_VARARGS,
        "Try returning a numpy array from C"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initqdbc(void) {
    TRACE("Initializing module");
    PyObject* m;
    m = Py_InitModule("qdbc", QdbcMethods);
    
    if(m == NULL) {
        return;
    }

    QdbcError = PyErr_NewException("qdbc.error", NULL, NULL);
    Py_INCREF(QdbcError);
    PyModule_AddObject(m, "error", QdbcError);

    PyDateTime_IMPORT;
    import_array();
}

int main(int argc, char** argv) {
    Py_SetProgramName(argv[0]);
    Py_Initialize();
    initqdbc();

    return 0;
}

