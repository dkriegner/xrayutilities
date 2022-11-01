# This file is part of xrayutilities.
#
# xrayutilies is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2009 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2010-2021 Dominik Kriegner <dominik.kriegner@gmail.com>

import glob
import os.path
import subprocess
import sys
import tempfile

import numpy
import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


def has_flag(compiler, flagname, output_dir=None):
    # see https://bugs.python.org/issue26689
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile('w', suffix='.c', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        f.close()
        try:
            obj = compiler.compile([f.name], output_dir=output_dir,
                                   extra_postargs=[flagname])
            os.remove(*obj)
        except setuptools.distutils.errors.CompileError:
            return False
        finally:
            os.remove(f.name)
    return True


class build_ext_subclass(build_ext):
    description = "Builds the Python C-extension of xrayutilities"
    user_options = (
        build_ext.user_options +
        [('without-openmp', None, 'build without OpenMP support')])

    def initialize_options(self):
        super().initialize_options()
        self.without_openmp = 0

    def finalize_options(self):
        super().finalize_options()
        self.without_openmp = int(self.without_openmp)

    def build_extensions(self):
        c = self.compiler.compiler_type
        # set extra compiler options
        copt = {}

        if c in copt:
            for flag in copt[c]:
                if has_flag(self.compiler, flag, self.build_temp):
                    for e in self.extensions:
                        e.extra_compile_args.append(flag)

        # set openMP compiler options
        if not self.without_openmp:
            openmpflags = {'msvc': ('/openmp', None),
                           'mingw32': ('-fopenmp', '-fopenmp'),
                           'unix': ('-fopenmp', '-lgomp')}
            if c in openmpflags:
                flag, lib = openmpflags[c]
                if has_flag(self.compiler, flag, self.build_temp):
                    for e in self.extensions:
                        e.extra_compile_args.append(flag)
                        if lib is not None:
                            e.extra_link_args.append(lib)
                        e.define_macros.append(('__OPENMP__', None))

        super().build_extensions()


class build_with_database(build_py):
    def build_database(self):
        dbfilename = os.path.join(self.build_lib, 'xrayutilities',
                                  'materials', 'data', 'elements.db')
        cmd = [sys.executable,
               os.path.join('lib', 'xrayutilities', 'materials',
                            '_create_database.py'),
               dbfilename]
        self.mkpath(os.path.dirname(dbfilename))
        print('building database: {}'.format(dbfilename))
        try:
            if sys.version_info >= (3, 5):
                subprocess.run(cmd, stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE, check=True)
            else:
                subprocess.check_output(cmd)
        except subprocess.CalledProcessError as cpe:
            sys.stdout.buffer.write(cpe.stdout)
            sys.stdout.buffer.write(cpe.stderr)
            raise

    def run(self):
        super().run()
        self.build_database()


cmdclass = {'build_py': build_with_database,
            'build_ext': build_ext_subclass}

with open('README.md') as f:
    long_description = f.read()

extmodul = Extension('xrayutilities.cxrayutilities',
                     sources=glob.glob(os.path.join('src', '*.c')))

with open('lib/xrayutilities/VERSION') as version_file:
    version = version_file.read().strip().replace('\n', '.')

setup(
    name="xrayutilities",
    version=version,
    author="Eugen Wintersberger, Dominik Kriegner",
    description="package for x-ray diffraction data evaluation",
    classifiers=[
        "Programming Language :: C",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License v2 or later "
        "(GPLv2+)"
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="eugen.wintersberger@desy.de, dominik.kriegner@gmail.com",
    maintainer="Dominik Kriegner",
    maintainer_email="dominik.kriegner@gmail.com",
    package_dir={'': 'lib'},
    packages=find_packages('lib'),
    package_data={
        "xrayutilities": ["VERSION", "*.conf"],
        "xrayutilities.materials": [os.path.join("data", "*")]
    },
    python_requires='~=3.6',
    setup_requires=['numpy', 'scipy', 'h5py'],
    install_requires=['numpy>=1.9.2', 'scipy>=0.18.0', 'h5py'],
    extras_require={
        'plot': ["matplotlib>=3.1.0"],
        'fit': ["lmfit>=1.0.1"],
        '3D': ["mayavi"],
    },
    include_dirs=[numpy.get_include()],
    ext_modules=[extmodul],
    cmdclass=cmdclass,
    url="https://xrayutilities.sourceforge.io",
    license="GPLv2",
)
