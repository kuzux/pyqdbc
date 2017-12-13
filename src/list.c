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

PyObject* get_list(K k) {
    TRACE("getting list");

    /* list of length n = dimensions are (n,) */
    /* we're using a long instead of size_t here, as GCC complains when we use an
     * unsigned integer instead of a signed one. I have no idea why numpy requires
     * a signed integer for a length. */
    long len = k->n;
    long num_dims = 1;

    int numpy_arr_type = 0;

    if(len < 0) {
        TRACE("wtf len is negative %ld", len);
        PyErr_SetString(QdbcError, "List length negative");
        return NULL;
    }

    switch(k->t) {
        /* numpy can handle arrays of arbitrary python objects. There's no 
         * guarantee that it's going to be fast, though :) */
        case 0: numpy_arr_type = NPY_OBJECT; break;    /* mixed list */
        case 1: numpy_arr_type = NPY_BOOL; break;      /* bool */
        case 4: numpy_arr_type = NPY_BYTE; break;      /* byte */
        case 5: numpy_arr_type = NPY_SHORT; break;     /* short */
        case 6: numpy_arr_type = NPY_INT32; break;     /* int */
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
        case 0: /* mixed */
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
        case 1: /* bool */
            {
                bool* dest = PyArray_GETPTR1(array, 0);
                /* G is unsigned char, we can convert it to bool */
                bool* src = (bool*)kG(k);
                memcpy(dest, src, len*sizeof(bool));
            }
            ret_val = (PyObject*)array;
            break;
        case 4: /* byte */ 
            {
                uint8_t* dest = PyArray_GETPTR1(array, 0);
                uint8_t* src = kG(k);
                memcpy(dest, src, len*sizeof(uint8_t));
            }
            ret_val = (PyObject*)array;
            break;
        case 5: /* short */
            {
                int16_t* dest = PyArray_GETPTR1(array, 0);
                int16_t* src = kH(k);
                memcpy(dest, src, len*sizeof(int16_t));
            }
            ret_val = (PyObject*)array;
            break;
        case 6: /* int */
            {
                /* Q int values are always 32 bit, so we use int32_t */
                int32_t* dest = PyArray_GETPTR1(array, 0);
                int32_t* src = kI(k);
                memcpy(dest, src, len*sizeof(int32_t));
            }
            ret_val = (PyObject*)array;
            break;
        case 7: /* long */
            {
                int64_t* dest = PyArray_GETPTR1(array, 0);
                /* not sure why J(long long) is incompatible with int64_t(also long long) */ 
                int64_t* src = (int64_t*)kJ(k);
                memcpy(dest, src, len*sizeof(int64_t));
            }
            ret_val = (PyObject*)array;
            break;
        case 8: /* real */
            {
                float* dest = PyArray_GETPTR1(array, 0);
                float* src = kE(k);
                memcpy(dest, src, len*sizeof(float));
            }
            ret_val = (PyObject*)array;
            break;
        case 9: /* float */
            {
                double* dest = PyArray_GETPTR1(array, 0);
                double* src = kF(k);
                memcpy(dest, src, len*sizeof(double));
            }

            ret_val = (PyObject*)array;
            break;
        case 10: /* char */
            /* TODO: How do we implement that? Char arrays are strings in Q as well */
            break;
        case 11: /* symbol */
            TRACE("Getting a list of symbols");
            /* Creaing a list of strings fails, so we need to create it with a
             * longer method here */
            if(array != NULL) {
                TRACE("wtf how did this get created");
            }

            {
                PyArray_Descr* dtype = PyArray_DescrNewFromType(NPY_STRING);

                TRACE("%d", dtype->type_num);
                if(dtype->subarray != NULL) TRACE("subarray");
                if(dtype->fields != NULL) TRACE("fields");

            }

            {
                for(long i=0; i<len; i++) {
                    char** dest = PyArray_GETPTR1(array, 0);
                    char* src = kS(k)[i];
                    TRACE("%s", src);
                    strncpy(*dest, src, SYMBOL_MAXLEN);
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

