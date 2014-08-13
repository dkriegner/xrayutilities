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
# Copyright (C) 2013 Dominik Kriegner <dominik.kriegner@gmail.com>


"""
convenience functions to open files for various data file reader

these functions should be used in new parsers since they transparently allow to open gzipped and bzipped files
"""

import os
import gzip
import bz2
import sys

if sys.version_info >= (3,3):
    import lzma # new in python 3.3

from .. import config
from ..exception import InputError

def xu_open(filename,mode='rb'):
    """
    function to open a file no matter if zipped or not. Files with extension
    '.gz' or '.bz2' are assumed to be compressed and transparently opened to read like 
    usual files.

    Parameters
    ----------
     filename:  filename of the file to open (full including path)
     mode:      mode in which the file should be opened
    
    Returns
    -------
     file handle of the opened file

    If the file does not exist an IOError is raised by the open routine, which is not
    caught within the function
    """

    if os.path.splitext(filename)[-1] == '.gz':
        fid = gzip.open(filename,mode)
    elif os.path.splitext(filename)[-1] == '.bz2':
        fid = bz2.BZ2File(filename,mode)
    elif os.path.splitext(filename)[-1] == '.xz':
        if sys.version_info >= (3,3):
            fid = lzma.open(filename,mode)
        else:
            raise TypeError("File compression type not supported in python versions prior to 3.3")
    else:
        fid = open(filename,mode)

    return fid

