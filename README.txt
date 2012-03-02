
xrayutilities - a package with useful scripts for X-ray diffraction
===================================================================

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
Installing xrayutilities is a two step process
1.) installing required third party software
    requirements are:
     scons (pythonic build system)
     C-compiler
     HDF5
     pytables
     scipy
     numpy
     matplotlib (optionally)
    refer to your operating system documentation to find out how to install
    those packages.
2.) build and install the C-library libxrutils.so/xrutils.dll, as well as the 
    python package (xrutils)

Obtaining the source code
=========================

The sources are hosted on sourceforge in git repository.
Use:
 $> git clone git://git.code.sf.net/p/xrayutilities/code xrayutilities
to clone the git repository. If you would like to have commit rights 
contact one of the administrators.

Building and installing the C library and python package
========================================================

Open a terminal and navigate to the source folder of xrayutilities.
xrayutilities use SCons to build and install C code. Installation
of libxrutils.so and the python package requires two steps
-> compile the library by simply typing 
    $>scons
   or
    $>scons debug=1
   to build with "-g -O0"
-> install the library and python package, either system wide 
    $>scons install
   , which means in /usr/lib/ on Unix systems.
   or locally in the user directory
    $>scons install --prefix=<path to install directory>
-> the documentation can be built with
    $>scons doc

The --prefix option sets the root directory for the installation.
Tools are installed under <prefix>/bin the library under
<prefix>/lib. 
If you use a package manager the SConstruct file includes support for DESTDIR.
To use this feature call

    $>scons DESTDIR=/destdir_path install 

instead of the command given above. This can be used in combination with the
prefix flag. 

The python package configuration
================================

The following steps should only be necessary for user local installation to
ensure the python module is found by the python interpreter:
In this case the module is installed under 
<prefix>/lib[64]/python?.?/site-packages on Unix systems and
<prefix>/Lib/site-packages on Windows systems. 

If you have installed the Python package in a directory unknown to your 
local Python distribution, you have to tell Python where to look for the Package. There are several 
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

if you already installed xrayutilities you can update it by navigating into
its source folder and obtain the new sources by

  $> git pull

if any code changed during the update you need to reinstall the libary and
python package:

  $> scons install --prefix=<path to install directory>


DOCUMENTATION
=============

Documention for xrayutilities is found in the doc folder. The manual (pdf)
can be built using scons

  $> scons doc
 
The API-documentation can be browsed by 

  $> pydoc -p PORT
 
in any web-browser, after the installation is finished.


PACKAGING
=========

create a tarball for redistribution of xrayutilities without the use of SVN

  $>scons dist

creates a tarball in the directory dist, which contains everything needed for
the installation of xrayutilities
