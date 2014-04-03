from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'bodymodel',
  ext_modules = cythonize("body_model.pyx"),
)