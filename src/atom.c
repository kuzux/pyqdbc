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

PyObject* get_atom(K k) {
    TRACE("reading atom");

    PyObject* tmp;

    /* Basic Q datatypes:
     *   (taken from 'Q for mortals', don't think it's accurate as far as ints 
     *   and longs are concerned 
     * 1b      -> boolean
     * 0x26    -> byte (8bit)
     * 42h     -> short (16bit)
     * 42      -> int (32bit)
     * 42j     -> long (64bit)
     * 4.2e    -> real (single precision)
     * 4.2f    -> float (double precision)
     * "c"     -> char (not sure if unicode)
     * `zaphod -> symbol (definitely not a string). python, unfortunately,
     *   lacks a symbol type
     * ========================
     * temporal types: (python is a bit lacking in this regard as well)
     * 2015.01.01T00:00:00.000000000 -> timestamp (explained in detail
     *   below)
     * 2006.07m   -> month
     * 2006.07.21 -> date
     * 2006.07.21T09:13:39 -> datetime (how common are those?)
     * 12:00:00.000000000 -> timespan
     * 23:59        -> minute
     * 23:59:59     -> second
     * 09:01:02:042 -> time
     * =====================
     * :: -> void
     */
    switch(k->t) {
        case -1: return PyBool_FromLong(k->g);                        /* bool  */
        case -4: return PyInt_FromLong(k->g);                         /* byte  */ 
        case -5: return PyInt_FromLong(k->h);                         /* short */
        case -6: return PyInt_FromLong(k->i);                         /* int   */
        case -7: return PyLong_FromLong(k->j);                        /* long  */
        case -8: return PyFloat_FromDouble(k->e);                     /* real  */
        case -9: return PyFloat_FromDouble(k->f);                     /* float */
        case -10: return PyString_FromStringAndSize((char*)&k->i, 1); /* char */
        case -11: return PyString_FromString(k->s)  ;                 /* symbol */
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

