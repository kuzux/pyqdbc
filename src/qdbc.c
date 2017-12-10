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

#define KXVER 3
#include <k.h>

/* we still need this outside debug mode because we need to use snprintf */
#include <stdio.h>
#include <math.h>
/* Can we definitely work on windows? 
 * TODO: fix the build script on windows
 */
#include <fcntl.h>
#include <errno.h>

#define BUFSIZE 255

#define DEBUG
#ifdef DEBUG
    #define TRACE printf
#else
    #define printf
#endif

#ifdef _WIN32 /* Windows-related shit, basically */
  #define F_GETFL 3
  void __security_check_cookie(int i) {};
  int fcntl (intptr_t fd, int cmd, ...) { return 1; }
#endif

#define DATE_OFFSET 10957
#define DAYS_TO_SECS 86400
#define EPOCH_OFFSET_SECS 946677600

static PyObject* QdbcError;

PyObject* get_atom(K k) {
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
            return NULL; 
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
    return NULL;
}

static PyObject* get_dict(K k) {
    /* function stub */
    return NULL;
}

static PyObject* get_table(K k) {
    /* function stub */
    return NULL;
}

static PyObject* qdbc_query(PyObject* self, PyObject* args) {
    char* host;
    int port;
    char* query;
    int query_len;

    if(!PyArg_ParseTuple(args, "sis#"/* atma oc */, &host, &port, &query, &query_len)) {
        return NULL;
    }

    if(query_len == 0) {
        /* return an empty object if the query is simply empty */
        return Py_BuildValue("");
    }

    int conn;
    conn = khp(host, port);
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
        return NULL;
    }

    if (res->t == 101) {
        /* we have a void object */
        Py_INCREF(Py_None);
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
        retVal = get_list(res);
    }

    if(!retVal) {
        PyErr_SetString(QdbcError, "Deserializing this type is not yet implemented");
        return NULL;
    }
    
    kclose(conn);

    return retVal;
}

/* TODO: Write a check_connection helper */
/* TODO: Write qopen/qclose type functions to keep a single session open */

static PyMethodDef QdbcMethods[] = {
    {"query", qdbc_query, METH_VARARGS,
        "Run a KDB query"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initqdbc(void) {
    PyObject* m;
    m = Py_InitModule("qdbc", QdbcMethods);
    
    if(m == NULL) {
        return;
    }

    QdbcError = PyErr_NewException("qdbc.error", NULL, NULL);
    Py_INCREF(QdbcError);
    PyModule_AddObject(m, "error", QdbcError);

    PyDateTime_IMPORT;
}

int main(int argc, char** argv) {
    Py_SetProgramName(argv[0]);
    Py_Initialize();
    initqdbc();

    return 0;
}

