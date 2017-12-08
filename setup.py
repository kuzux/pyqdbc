from distutils.core import setup, Extension

module1 = Extension('qdbc',
                    sources = ['src/qdbc.c'],
                    include_dirs = ['include'],
                    library_dirs = ['lib'],
                    extra_objects = ['lib/c.o'])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])

