xrayutilities unittests
=======================

To run ALL the unittests execute in the package root

    python setup.py test

or 

    python -m unittest discover


Individual tests can be run by

    python -m unitest <testfile>


Additional test data
--------------------

To run specific tests (file parsers) additional test data files are needed.
If those files are not present the tests are skipped.  Because of their size
they are shipped seperately and can be downloaded at 
https://sourceforge.net/projects/xrayutilities/files/

The latest 'testdata' tarball has to be unpacked and placed in the
'tests/data' directory. 
