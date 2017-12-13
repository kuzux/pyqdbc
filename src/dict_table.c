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

PyObject* get_dict(K k) {
    TRACE("getting dictionary");

    K keys = kK(k)[0]; /* OK, how racist are we? */
    K data = kK(k)[1]; /* It's not getting any better */

    char** key_names = kS(keys);

    int num_keys = keys->n;
    TRACE("num_keys %d", num_keys);

    PyObject* ret_val = PyDict_New();

    for(int i=0;i<num_keys;i++) {
        TRACE("%s", key_names[i]);
        K item = kK(data)[i];
        TRACE("got item");
        if(item->t < 0) {
            PyDict_SetItemString(ret_val, key_names[i], get_atom(item));
        } else {
            PyDict_SetItemString(ret_val, key_names[i], get_list(item));
        }
    }

    return ret_val;
}

PyObject* get_table(K k) {
    /* function stub */
    TRACE("getting table");
    return NULL;
}

