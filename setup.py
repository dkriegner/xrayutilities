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
# Copyright (C) 2010-2019 Dominik Kriegner <dominik.kriegner@gmail.com>

import glob
import os.path
import subprocess
import sys
from distutils.command.build import build
from distutils.command.install import INSTALL_SCHEMES
from distutils.errors import DistutilsArgError
from distutils.fancy_getopt import FancyGetopt

import numpy
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


cliopts = []
cliopts.append(("without-openmp", None, "build without OpenMP support"))

options = FancyGetopt(option_table=cliopts)

# Modify the data install dir to match the source install dir
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

# first read all the arguments passed to the script
# we need to do this otherwise the --help commands would not work
args = sys.argv[1:]
try:
    # search the arguments for options we would like to use
    # get new args with the custom options stripped away
    args, opts = options.getopt(args)
except DistutilsArgError:
    pass

# set default flags
without_openmp = False

for opts, values in options.get_option_order():
    if opts == "without-openmp":
        without_openmp = True

copt = {'msvc': [],
        'mingw32': ['-std=c99'],
        'unix': ['-std=c99']}
lopt = {'mingw32': [],
        'unix': []}

user_macros = []
if not without_openmp:
    user_macros = [('__OPENMP__', None)]
    copt["msvc"].append('/openmp')
    copt["mingw32"].append("-fopenmp")
    copt["unix"].append("-fopenmp")
    lopt["mingw32"].append("-fopenmp")
    lopt["unix"].append("-lgomp")


class build_ext_subclass(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        # set custom compiler options
        if c in copt:
            for e in self.extensions:
                e.extra_compile_args = copt[c]
        if c in lopt:
            for e in self.extensions:
                e.extra_link_args = lopt[c]
        build_ext.build_extensions(self)


class build_with_database(build):
    def build_database(self):
        dbfilename = os.path.join(self.build_platlib, 'xrayutilities',
                                  'materials', 'data', 'elements.db')
        cmd = [sys.executable,
               os.path.join('lib', 'xrayutilities', 'materials',
                            '_create_database.py'),
               dbfilename]
        self.mkpath(os.path.dirname(dbfilename))
        try:
            if sys.version_info >= (3, 5):
                cp = subprocess.run(cmd, stderr=subprocess.PIPE,
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


cmdclass = {'build_ext': build_ext_subclass,
            'build': build_with_database}

with open('README.md') as f:
    long_description = f.read()

extmodul = Extension(
    'xrayutilities.cxrayutilities',
    sources=glob.glob(os.path.join('src', '*.c')),
    define_macros=user_macros
    )

with open('VERSION') as version_file:
    version = version_file.read().strip()

try:
    import sphinx
    from sphinx.setup_command import BuildDoc

    class build_doc(BuildDoc):
        def run(self):
            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version
            build = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build.build_lib))
            try:
                sphinx.setup_command.BuildDoc.run(self)
            except UnicodeDecodeError:
                print("ERROR: unable to build documentation"
                      " because Sphinx do not handle"
                      " source path with non-ASCII characters. Please"
                      " try to move the source package to another"
                      " location (path with *only* ASCII characters)")
            sys.path.pop(0)

    cmdclass['build_doc'] = build_doc
except ImportError:
    pass

setup(
    name="xrayutilities",
    version=version,
    author="Eugen Wintersberger, Dominik Kriegner",
    description="package for x-ray diffraction data evaluation",
    classifiers=[
        "Programming Language :: C",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License v2 or later "
        "(GPLv2+)"
        ],
    long_description=long_description,
    author_email="eugen.wintersberger@desy.de, dominik.kriegner@gmail.com",
    maintainer="Dominik Kriegner",
    maintainer_email="dominik.kriegner@gmail.com",
    package_dir={'': 'lib'},
    packages=find_packages('lib'),
    package_data={
        "xrayutilities": ["*.conf"],
        "xrayutilities.materials": [os.path.join("data", "*")]
        },
    data_files=[('xrayutilities', ['VERSION'])],
    python_requires='~=3.3',
    install_requires=['numpy>=1.9.2', 'scipy>=0.11.0', 'h5py', 'setuptools'],
    extras_require={
        'plot': ["matplotlib"],
        'fit': ["lmfit"],
        'lzma': ["lzma"],
        },
    include_dirs=[numpy.get_include()],
    ext_modules=[extmodul],
    cmdclass=cmdclass,
    url="http://xrayutilities.sourceforge.net",
    license="GPLv2",
    script_args=args
    )
