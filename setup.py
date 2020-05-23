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
import tempfile
from distutils.command.build_py import build_py
from distutils.command.install import INSTALL_SCHEMES
from distutils.errors import CompileError, DistutilsArgError
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

# get options from command line
without_openmp = False
for opts, values in options.get_option_order():
    if opts == "without-openmp":
        without_openmp = True


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
        except CompileError:
            return False
        finally:
            os.remove(f.name)
    return True


class build_ext_subclass(build_ext):

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
        if not without_openmp:
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
                        e.define_macros .append(('__OPENMP__', None))

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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
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
    install_requires=['numpy>=1.9.2', 'scipy>=0.11.0', 'h5py', 'setuptools'],
    extras_require={
        'plot': ["matplotlib"],
        'fit': ["lmfit>=1.0.1"],
        'lzma': ["lzma"],
        },
    include_dirs=[numpy.get_include()],
    ext_modules=[extmodul],
    cmdclass=cmdclass,
    url="http://xrayutilities.sourceforge.net",
    license="GPLv2",
    script_args=args
    )
