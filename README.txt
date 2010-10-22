Directories:
doc ........ directory for documentation
matlab ..... directory with matlab scripts and mex extensions
python ..... holds an installable python module 
src ........ source directory for the C-library
tools ...... sources and binaries for tools (executable programs)

INSTALLATION
============
Installing xrutils is a two step process
1.) installing required third party software
1.) build and install the C-library libxrutils.so
2.) install the python module

Obtaining the source code
-------------------------

So far this is only possible using the svn repository on brewster, ask Eugen
for information how to access the source code. 
No tarball releases were done so far.

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
>python setup.py install
will install the python module in the standard directory where Python looks for 
third party modules.
For user local installation use
>python setup.py install --home=<install path>
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
  
 $>scons
 $>scons install --prefix=<path to install directory>

