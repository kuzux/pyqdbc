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

#ifndef _QDBC_H
#define _QDBC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Python.h _might_ affect the behavior of other includes (not sure how) so it
 * needs to be included at the top of the file */
/* TODO: Try to make this python-version agnostic,
 * we don't depend on anything specifically in python2 */
#include <python2.7/Python.h>
/* Apparently datetimes in python need to be imported and initialized separately */
#include <python2.7/datetime.h>

/* TODO: Should we make the numpy dependency optional and fall back to classic
 * python lists? Or we can provide a pure-python implementation that uses Q's
 * HTTP support but is limited to 2000 rows.
 *
 * Kind of like this gist: 
 * https://gist.github.com/kuzux/e857e9633d780a14c836a7cdbcea9cef 
 * but with HTTP requests instead of spawning a Q process */
/* These defines are required by numpy for some reason */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL qdbc
#include <python2.7/numpy/arrayobject.h>

#define KXVER 3
#include <k.h>

/* we still need this outside debug mode because we need to use snprintf */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Can we definitely work on windows? 
 * TODO: fix the build script on windows
 * Also TODO: Test the build on windows and OSX
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

#define SYMBOL_MAXLEN 80

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

/* TODO: document the accessor functions here */

extern PyObject* QdbcError;

/* functions private to the module */
PyObject* get_atom(K k);
PyObject* get_list(K k);
PyObject* get_dict(K k);
PyObject* get_table(K k);

/* functions exported by the module */
PyObject* qdbc_query(PyObject* self, PyObject* args);
PyObject* qdbc_ping(PyObject* self, PyObject* args);

#endif

