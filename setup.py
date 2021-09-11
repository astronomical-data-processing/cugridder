
import os
from setuptools import setup, Extension
from distutils.command.install import install as _install

import sys


class _deferred_pybind11_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

include_dirs = ['./include', _deferred_pybind11_include(True),
                _deferred_pybind11_include()]
extra_compile_args = ['-Wall', '-Wextra', '-Wfatal-errors', '-Wstrict-aliasing=2', '-Wwrite-strings', '-Wredundant-decls', '-Woverloaded-virtual', '-Wcast-qual', '-Wcast-align', '-Wpointer-arith', '-Wfloat-conversion']
#, '-Wsign-conversion', '-Wconversion'
python_module_link_args = []

if sys.platform == 'darwin':
    import distutils.sysconfig
    extra_compile_args += ['--std=c++11', '--stdlib=libc++', '-mmacosx-version-min=10.9']
    vars = distutils.sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '')
    python_module_link_args+=['-bundle']
else:
    extra_compile_args += ['--std=c++11', '-march=native', '-O3', '-ffast-math']
    python_module_link_args += ['-march=native', '-Wl,-rpath,$ORIGIN']

def get_extension_modules():
    return [Extension('cuda_gridder',
                      sources=['src/cudagridder.cc'],
                      depends=['src/cudagridder.h',
                               'setup.py'],
                      include_dirs=include_dirs,
                      extra_compile_args=extra_compile_args,
                      extra_link_args=python_module_link_args)]

setup(name='cuda_gridder',
      version='0.1',
      description='Cuda Gridding/Degridding helper library for NIFTy',
      include_package_data=True,
      packages=[],
      setup_requires=['numpy>=1.15.0', 'pybind11>=2.2.4'],
      ext_modules=get_extension_modules(),
      install_requires=['numpy>=1.15.0', 'pybind11>=2.2.4']
      )
