from setuptools import setup
from setuptools import find_packages


setup(name='db-reader',
      version='0.1',
      description='Valeo\'s library to open, parse and decode recordings',
      author='Valeo.ai',
      install_requires=['cantools', 'numpy', 'pathlib','pandas'],
      packages=find_packages())
