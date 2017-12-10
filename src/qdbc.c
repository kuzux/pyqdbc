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

/* TODO: Should we make the numpy dependency optional and fall back to classic
 * python lists? Or we can provide a pure-python implementation that uses Q's
 * HTTP support but is limited to 2000 rows.
 *
 * Kind of like this gist: 
 * https://gist.github.com/kuzux/e857e9633d780a14c836a7cdbcea9cef 
 * but with HTTP requests instead of spawning a Q process */
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

    /* TODO: maybe we should add an example for each of Q's atom types? */
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
            /* The quirk of those objects is that the epoch is the millenium.
             * Jan 01, 2000. That's why we need all those offsets. To convert 
             * between the Q epoch and the unix epoch. This was probably done 
             * in Q as timestamp objs have ns precision and a lot of the modern
             * times would overflow, even on int64 if unix epoch was used */
            /* Update: If we count the nanoseconds from unix epoch, signed 64
             * bit integers overflow in the year 2400 something. Let's hope that
             * Q's choice of Epoch was because Arthur Whitney imaginedthe  30 years 
             * after the date I've roughly calculated was more important than 
             * compatibility with practically every other system in the world
             * and not simply because of wanting to be 'different'.
             * -- t3h PeNgU1N oF d00m */
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

static PyObject* get_list(K k) {
    /* function stub */
    TRACE("getting list");

    /* list of length n = dimensions are (n,) */
    /* we're using a long instead of size_t here, as GCC complains when we use an
     * unsigned integer instead of a signed one. I have no idea why numpy requires
     * a signed integer for a length. */
    long len = k->n;
    long num_dims = 1;

    int numpy_arr_type = 0;

    switch(k->t) {
        /* numpy can handle arrays of arbitrary python objects. There's no 
         * guarantee that it's going to be fast, though :) */
        case 0: numpy_arr_type = NPY_OBJECT; break;    /* mixed list */
        case 1: numpy_arr_type = NPY_BOOL; break;      /* bool */
        case 4: numpy_arr_type = NPY_BYTE; break;      /* byte */
        case 5: numpy_arr_type = NPY_SHORT; break;     /* short */
        case 6: numpy_arr_type = NPY_INT; break;       /* int */
        case 7: numpy_arr_type = NPY_LONG; break;      /* long */
        case 8: numpy_arr_type = NPY_FLOAT; break;     /* real */
        case 9: numpy_arr_type = NPY_DOUBLE; break;    /* float */
        case 10: numpy_arr_type = NPY_UNICODE; break;  /* char */
        case 11:                                       /* symbol */
            /* Neither python nor numpy has a concept of symbols, with
             * the lookup table and all. That would be a significant advantage.
             * Ruby is definitely better in this regard. Maybe we should return
             * a lookup table and an array of integers in this case? (For
             * performance reasons, I guess). For the case of returning a
             * single symbol, converting it to a string is fine, though */
            numpy_arr_type = NPY_STRING; 
            break;
        /* all the temporal types */
        case 12:                                       /* timestamp */
        case 13:                                       /* month */
        case 14:                                       /* date */
        case 15:                                       /* datetime */
            numpy_arr_type = NPY_DATETIME;
            break;
        case 16:                                       /* timespan */
        case 17:                                       /* minute */
        case 18:                                       /* second */
        case 19:                                       /* time */
            numpy_arr_type = NPY_TIMEDELTA;
            break;
            /* Yup, there's no hour type. I can see how it is not needed,
             * but given that we have that many datetime types, including 
             * an hour type for orthogonality's sake wouldn't be too weird. */
        default:
            PyErr_SetString(QdbcError, "Unsupported array type");
            return NULL;
    }
    
    /* I think I've written k->t too many times and now this song is playing
     * in my head: https://youtu.be/FGT0A2Hz-uk */
    /* Plus; why does my code read like a piece of stream-of-consciousness
     * writing? */
    
    /* PyObject and PyArrayObject's representations are compatible with each
     * other, so we can safely cast between those two types */
    /* Also; the pointer for the argument of this function is only needed to
     * be able to get an array, the value is copied within the function,
     * and I'm too lazy to initialize a new array of length one, so I am
     * sending in a pointer to a stack value here. Think of it as a stack
     * allocated array of length 1 */
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(num_dims, &len, numpy_arr_type);
    PyObject* ret_val = NULL;

    switch(k->t) {
        case 0:
            {
                for(size_t i=0;i<len;i++) {
                    K elem = kK(k)[i];
                    if(elem->t<0) {
                        /* This doesn't work without that temp variable as that expression is
                         * not an lvalue */
                        PyObject** addr = PyArray_GETPTR1(array,i);
                        *addr = get_atom(elem);
                    } else {
                        /* TODO: Implement this */
                        PyErr_SetString(QdbcError, "Getting nested lists not yet implemented");
                        return NULL;
                    }
                }
            }
            ret_val = (PyObject*)array;
            break;
        default:
            TRACE("list parsing not yet implemented for %d", k->t);
            break;
    }

    if(ret_val == NULL) {
        PyErr_SetString(QdbcError, "Couldn't parse list");
        return NULL;
    }

    return ret_val;
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
    } else if(res->t == 98) {
        retVal = get_table(res);
    } else if(res->t == 99) {
        retVal = get_dict(res);
    } else {
        /* This handles mixed lists as well */
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

