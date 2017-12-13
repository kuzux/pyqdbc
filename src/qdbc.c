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

#include <qdbc.h>

PyObject* QdbcError;

PyObject* qdbc_query(PyObject* self, PyObject* args) {
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

    TRACE("Object type %d", res->t);

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

    if(!retVal /*Check if we got a null value */ && 
            !PyErr_Occurred() /* Check we haven't received a previous exception */ ) {
        PyErr_SetString(QdbcError, "Deserializing this type is not yet implemented");
        return NULL;
    }

    TRACE("closing connection");
    kclose(conn);

    return retVal;
}

PyObject* qdbc_ping(PyObject* self, PyObject* args) {
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
    {NULL, NULL, 0, NULL}
};

/* name of this function is actually important, this is the main
 * init method called by the python interpreter */
PyMODINIT_FUNC initqdbc(void) {
    TRACE("Initializing module");

    PyObject* m;
    m = Py_InitModule("qdbc", QdbcMethods);
    
    if(m == NULL) {
        TRACE("Module initialization failed");
        return;
    }

    QdbcError = PyErr_NewException("qdbc.error", NULL, NULL);
    Py_INCREF(QdbcError);
    PyModule_AddObject(m, "error", QdbcError);

    PyDateTime_IMPORT;
    import_array();
    TRACE("init succeeded");
}

/* We still need a main function in the extension code to call if we're
 * directly loaded by the python interpreter instead of being import'ed */
int main(int argc, char** argv) {
    Py_SetProgramName(argv[0]);
    Py_Initialize();
    initqdbc();

    return 0;
}

