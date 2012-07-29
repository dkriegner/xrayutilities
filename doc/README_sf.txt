
xrayutilities - a package with useful scripts for X-ray diffraction
===================================================================

 Copyright (C) 2009-2012 Eugen Wintersberger <eugen.wintersberger@desy.de>
 Copyright (C) 2009-2012 Dominik Kriegner <dominik.kriegner@gmail.com>


Obtaining the source code
=========================

The sources are hosted on sourceforge in git repository and are made
available as tarballs from time to time. 
Download the latest tarball or use:
 $> git clone git://git.code.sf.net/p/xrayutilities/code xrayutilities
to clone the git repository. If you would like to have commit rights 
contact one of the administrators.



INSTALLATION
============
Installing xrayutilities is a two step process. A few notes on how to install
on the various can be found in the README shipped with the package's sources.
For installation on Windows have a look to the documentation.

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





