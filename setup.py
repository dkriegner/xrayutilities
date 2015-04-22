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
# Copyright (C) 2010-2011,2013 Dominik Kriegner <dominik.kriegner@gmail.com>

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.fancy_getopt import FancyGetopt
import os.path
import numpy
import sys

cliopts = []
cliopts.append(("without-openmp", None, "build without OpenMP support"))

options = FancyGetopt(option_table=cliopts)

# first read all the arguments passed to the script
# we need to do this otherwise the --help commands would not work
args = sys.argv[1:]
try:
    # search the arguments for options we would like to use
    # get new args with the custom options stripped away
    args, opts = options.getopt(args)
except:
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
        if c in list(copt.keys()):
            for e in self.extensions:
                e.extra_compile_args = copt[c]
        if c in list(lopt.keys()):
            for e in self.extensions:
                e.extra_link_args = lopt[c]
        build_ext.build_extensions(self)

cmdclass = {'build_ext': build_ext_subclass}

with open('README.txt') as f:
    long_description = f.read()

extmodul = Extension(
    'xrayutilities.cxrayutilities',
    sources=[os.path.join('xrayutilities', 'src', 'cxrayutilities.c'),
             os.path.join('xrayutilities', 'src', 'gridder_utils.c'),
             os.path.join('xrayutilities', 'src', 'gridder1d.c'),
             os.path.join('xrayutilities', 'src', 'gridder2d.c'),
             os.path.join('xrayutilities', 'src', 'block_average.c'),
             os.path.join('xrayutilities', 'src', 'qconversion.c'),
             os.path.join('xrayutilities', 'src', 'gridder3d.c'),
             os.path.join('xrayutilities', 'src', 'file_io.c')
             ],
    define_macros=user_macros
    )

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
    version="1.1.1",
    author="Eugen Wintersberger, Dominik Kriegner",
    description="package for x-ray diffraction data evaluation",
    classifiers=[
        "Programming Language :: C",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
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
    packages=[
        "xrayutilities", "xrayutilities.math", "xrayutilities.io",
        "xrayutilities.materials", "xrayutilities.analysis"
        ],
    package_data={
        "xrayutilities": ["*.conf"],
        "xrayutilities.materials": [
            os.path.join("data", "*.db"),
            os.path.join("data", "*.cif")
            ]
        },
    requires=['numpy', 'scipy', 'matplotlib', 'tables'],
    include_dirs=[numpy.get_include()],
    ext_modules=[extmodul],
    cmdclass=cmdclass,
    url="http://xrayutilities.sourceforge.net",
    license="GPLv2",
    script_args=args
    )
