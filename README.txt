Directories:
doc ........ directory for documentation
matlab ..... directory with matlab scripts and mex extensions
python ..... holds an installable python module 
src ........ source directory for the C-library
tools ...... sources and binaries for tools (executable programs)

INSTALLATION
============
Installing xrutils is a two step process
1.) build and install the C-library libxrutils.so
2.) install the python module

Building the library
--------------------
xrutils uses SCons to build and install C code. Installation
of libxrutils.so requires two steps
-> compile the library by simply typing 
   $>scons 
-> install the library and tools with 
   $>scons install --prefix=<path to install directory>

The --prefix option sets the root directory for the installation.
Tools are installed under <prefix>/bin the library under
<prefix>/lib. 
During the install run the Scons writes a confg.py file in 
./python/xrutils which contains the installation directory of the 
C library. This is necessary to make ctypes aware of the location 
where the library is installed. On Unix systems this step would 
not he necessary since one could use the LD_LIBRARY_PATH 
environment variable. However, on Windows systems no such facility 
is available.

Installing the Python module
----------------------------
