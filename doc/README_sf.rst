
xrayutilities
=============

**a package with useful scripts for X-ray diffraction**
 
 
Copyright (C) 2009-2016 Dominik Kriegner <dominik.kriegner@gmail.com>

Copyright (C) 2009-2013 Eugen Wintersberger <eugen.wintersberger@desy.de>


Obtaining the source code
=========================

The sources are hosted on sourceforge in git repository and are made
available as tarballs from time to time. 
Download the latest tarball or use:

    git clone git://git.code.sf.net/p/xrayutilities/code xrayutilities


to clone the git repository. If you would like to have commit rights 
contact one of the administrators.


Installation
============

Installing xrayutilities is a two step process. A few notes on how to install
on the various can be found in the README shipped with the package's sources.

1. installing required third party software
   requirements are:
   
   - C-compiler
   - h5py (for HDF5 file access)
   - scipy
   - numpy
   - matplotlib (optionally)
   
   refer to your operating system documentation to find out how to install
   those packages. On Windows we suggest to use Python(x,y)
    
2. install *xrayutilities* using distutils by executing

    python setup.py install


For details of how to setup your Python installation to find xrayutilities
after the installation please refer to the documention.

