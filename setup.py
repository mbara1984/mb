from distutils.extension import Extension
from Cython.Distutils import build_ext
from setuptools import setup, Extension, Command
from setuptools import setup

from distutils.extension import Extension
from setuptools import setup


setup(name='mb',
      version='0.10',
      description='geutils',
      url='none.yet',
      author='Maciej Baranski',
      author_email='baranskimaciej1984@gmail.com',
      license='unpublished-so-far',
      packages=['mb'],
      ext_modules = [],
      zip_safe=False)

# python setup.py develop --user
