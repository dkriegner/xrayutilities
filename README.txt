
xrutils - a package with useful scripts for X-ray diffraction physicists
========================================================================

 Copyright (C) 2009-2012 Eugen Wintersberger <eugen.wintersberger@desy.de>
 Copyright (C) 2009-2012 Dominik Kriegner <dominik.kriegner@aol.at>

Directories:
doc ........ directory for documentation
examples ... directory with example scripts and configurations
python ..... holds an installable python module
src ........ source directory for the C-library used by the python module
tools ...... sources and binaries for tools (executable programs)


INSTALLATION
============
Installing xrutils is a two step process
1.) installing required third party software
1.) build and install the C-library libxrutils.so
2.) install the python module

Obtaining the source code
-------------------------

The sources are hosted on sourceforge in git repository.
Use:
 $> git clone git://git.code.sf.net/p/xrayutilities/code xrayutilities
to clone the git repository. If you would like to have commit rights
contact one of the administrators.

Building and installing the C library
-------------------------------------

Open a terminal and navigate to the source folder of xrutils (trunk).
xrutils uses SCons to build and install C code. Installation
of libxrutils.so requires two steps
-> compile the library by simply typing
   $>scons
   or
   $>scons debug=1
   to build with "-g -O0"
-> install the library and tools with
   $>scons install --prefix=<path to install directory>
-> the documentation can be built with
   $>scons doc

The --prefix option sets the root directory for the installation.
Tools are installed under <prefix>/bin the library under
<prefix>/lib.

Installing the Python module
----------------------------
Tow possible installation procedures for the Python module are supported
1.) system wide installtion in the systems Python installation
2.) user local installation

In the first case a simple
 $> python setup.py install
will install the python module in the standard directory where Python looks for
third party modules.
For user local installation use
 $> python setup.py install --home=<install path>
In this case the module is installed under <install path>/lib/python.

If you have installed the Python package in a directory unknown to your
local Python distribution (usually the case for user local installation)
you have to tell Python where to look for the Package. There are several
ways how to do this:

-) add the directory where the package is installed to your
   PYTHONPATH environment variable.

-) add the path to sys.path in the .pythonrc file placed in your home
   directory.

   import sys
   sys.path.append("path to the xrutils package")

-) simply apply the previous method in every script where you want to
   use the xrutils package before importing the package:

   import sys
   sys.path.append("path to the xrutils package")
   import xrutils


UPDATE
======

if you already installed xrutils you can update it by navigating into its
source folder (trunk) and obtain the new sources by

 $> svn update

if only python files were updated you only need to perform the installation
of the python package using

 $> python setup.py install --home=<install path>

if any c-code changed during the update you also need to rebuild the c-library

 $> scons
 $> scons install --prefix=<path to install directory>


DOCUMENTATION
=============

Documention for xrutils is found in the doc folder. The manual (pdf) can be
built using scons

 $> scons doc

The API-documentation can be browsed by

 $> pydoc -p PORT

in any web-browser, after the installation is finished.


PACKAGING
=========

create a tarball for redistribution of xrutils without the use of SVN

 $>scons dist

creates a tarball in the directory dist, which contains everything needed for
the installation of xrutils
