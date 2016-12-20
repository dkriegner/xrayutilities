# This file is part of xrayutilities.
#
# xrayutilities is free software; you can redistribute it and/or modify
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
# Copyright (C) 2013-2016 Dominik Kriegner <dominik.kriegner@gmail.com>


"""
convenience functions to open files for various data file reader

these functions should be used in new parsers since they transparently allow to
open gzipped and bzipped files
"""

import gzip
import bz2
import sys
import h5py

from .. import config
from ..exception import InputError

if sys.version_info >= (3, 3):
    import lzma  # new in python 3.3


# python 2to3 compatibility
try:
    basestring
except NameError:
    basestring = str


def xu_open(filename, mode='rb'):
    """
    function to open a file no matter if zipped or not. Files with extension
    '.gz', '.bz2', and '.xz'  are assumed to be compressed and transparently
    opened to read like usual files.

    Parameters
    ----------
     filename:  filename of the file to open (full including path)
     mode:      mode in which the file should be opened

    Returns
    -------
     file handle of the opened file

    If the file does not exist an IOError is raised by the open routine, which
    is not caught within the function
    """

    if filename.endswith('.gz'):
        fid = gzip.open(filename, mode)
    elif filename.endswith('.bz2'):
        fid = bz2.BZ2File(filename, mode)
    elif filename.endswith('.xz'):
        if sys.version_info >= (3, 3):
            fid = lzma.open(filename, mode)
        else:
            try:
                import contextlib
                import lzma
                fid = contextlib.closing(lzma.LZMAFile(filename, mode))
            except:
                raise TypeError("File compression type not supported! Install "
                                "pyliblzma or switch to Python >3.3")
    else:
        fid = open(filename, mode)

    return fid


class xu_h5open(object):
    """
    helper object to decide if a HDF5 file has to be opened/closed when
    using with a 'with' statement.
    """
    def __init__(self, f, mode='r'):
        """
        Parameters
        ----------
         f:     filename or h5py.File instance
         mode:  mode in which the file should be opened. ignored in case a
                file handle is passed as f
        """
        self.closeFile = True
        self.fid = None
        self.mode = mode
        if isinstance(f, h5py.File):
            self.fid = f
            self.closeFile = False
            self.filename = f.filename
        elif isinstance(f, basestring):
            self.filename = f
        else:
            raise InputError("f argument of wrong type was passed, "
                             "should be string or filename")

    def __enter__(self):
        if self.fid:
            if not self.fid.fid.valid:
                self.fid = h5py.File(self.filename, self.mode)
        else:
            self.fid = h5py.File(self.filename, self.mode)
        return self.fid

    def __exit__(self, type, value, traceback):
        if self.closeFile:
            self.fid.close()
