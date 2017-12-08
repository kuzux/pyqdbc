from distutils.core import setup, Extension

qdbc = Extension('qdbc',
                    sources = ['src/qdbc.c'],
                    include_dirs = ['include'],
                    library_dirs = ['lib'],
                    extra_objects = ['lib/c.o'])

setup (name = 'PyQdbc',
       version = '0.0.1',
       description = 'kdb/q bindings for python',
       ext_modules = [qdbc])
