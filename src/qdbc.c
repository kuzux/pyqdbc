#include <Python.h>
#include <datetime.h>

#define KXVER 3
#include <k.h>

#include <fcntl.h>
#include <errno.h>
#include <stdio.h>

#define BUFSIZE 255

#ifdef _WIN32 /* Windows-related shit, basically */
  #define F_GETFL 3
  void __security_check_cookie(int i) {};
  int fcntl (intptr_t fd, int cmd, ...) { return 1; }
#endif

#define TIMESTAMP_OFFSET 946677600
#define DATETIME_OFFSET 
static PyObject* QdbcError;

PyObject* get_atom(K k) {
    long micros, secs;
    double us;

    int years;
    int months;
    int days;

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
            micros = (k->j%1000000000)/1000;
            secs = (k->j/1000000000);
            us = (micros/1000000.0);

            tmp = PyFloat_FromDouble(TIMESTAMP_OFFSET+secs+us);
            tmp = Py_BuildValue("(o)", tmp);
            return PyDateTime_FromTimestamp(tmp);
        case -13:
            /* month */
            months = 1 + (k->i)%12;
            years = 2000 + (k->i)/12;
            return PyDateTime_FromDateAndTime(years, months, 1, 0, 0, 0, 0);
        case -14:
            /* date */
            days = 1 + (k->i)%30;
            months = 1 + ((k->i)/30)%12;
            years = 2000 + ((k->i)/30*12);

            return PyDateTime_FromDateAndTime(years, months, days, 0, 0, 0, 0);
        case -15:
            /* datetime */
            return PyFloat_FromDouble(k->f);
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

    PyObject* retVal = NULL;

    if(res->t < 0) {
       retVal = get_atom(res);
    }

    if(!retVal) {
        PyErr_SetString(QdbcError, "Deserializing this type is not yet implemented");
        return NULL;
    }
    
    kclose(conn);

    return retVal;
}

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
}

int main(int argc, char** argv) {
    Py_SetProgramName(argv[0]);
    Py_Initialize();
    initqdbc();

    return 0;
}

