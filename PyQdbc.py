# Copyright (c) 2017 Arda Cinar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import qdbc

# TODO: write some code to package the whole thing

class QConnection(object):
    def __init__(self, host="localhost", port=20001):
        self.host = host
        self.port = port
    def query(self, query):
        qdbc.query(self.host, self.port, query)
    def get_table(self, query, conds={}, cols=[]):
        # selecting from a table using a few equals statements is super
        # common in partitioned table
        cond_str = ",".join(map(lambda pair: "%s=%s" % pair, conds.iteritems()))

        # an empty columns list defaults to selecting all columns in Q
        cols_str = ",".join(cols)

        query_str = "select %s from (%s) where %s" % (cols_str, query, cond_str)
        
        # TODO: pandas has no c api, so we need to convert from
        # a dict of numpy arrays to a dataframe here
        print query_str

        return None

