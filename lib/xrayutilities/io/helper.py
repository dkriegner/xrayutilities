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
# Copyright (c) 2013-2019, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>


"""
convenience functions to open files for various data file reader

these functions should be used in new parsers since they transparently allow to
open gzipped and bzipped files
"""

import bz2
import gzip
import io
import lzma
import string

from operator import itemgetter

import h5py

from .. import config
from ..exception import InputError


def generate_filenames(filetemplate, scannrs=None):
    """
    generate a list of filenames from a template and replacement values.

    Parameters
    ----------
    filetemplate: str or list
      template string which should contain placeholders if scannrs is not None
    scannrs: iterable, optional
      list of scan numbers. If None then the filetemplate will be returned.

    Examples
    --------
    >>> generate_filenames("filename_%d.ras", [1, 2, 3])
    ['filename_1.ras', 'filename_2.ras', 'filename_3.ras']

    >>> generate_filenames("filename_{}.ras", [1, 2, 3])
    ['filename_1.ras', 'filename_2.ras', 'filename_3.ras']

    >>> generate_filenames("filename_{}_{}.ras", [(11, 1), (21, 2), (31, 3)])
    ['filename_11_1.ras', 'filename_21_2.ras', 'filename_31_3.ras']

    >>> generate_filenames("filename_%d.ras", 1)
    ['filename_1.ras']

    >>> generate_filenames("filename.ras")
    ['filename.ras']

    >>> generate_filenames(["filename.ras", "othername.ras"])
    ['filename.ras', 'othername.ras']

    Returns
    -------
    list of filenames. If only a single filename is returned it will still be
    encapsulated in a list
    """
    if scannrs is None:
        if isinstance(filetemplate, list):
            return filetemplate
        return [filetemplate]

    files = []
    if not isinstance(scannrs, (list, tuple)):
        scannrs = [scannrs]
    placeholders = map(itemgetter(1), string.Formatter().parse(filetemplate))
    isformatstring = any(p is not None for p in placeholders)
    for nr in scannrs:
        if isinstance(nr, tuple) and isformatstring:
            files.append(filetemplate.format(*nr))
        elif isformatstring:
            files.append(filetemplate.format(nr))
        else:
            files.append(filetemplate % nr)

    return files


def xu_open(filename, mode='rb'):
    """
    function to open a file no matter if zipped or not. Files with extension
    '.gz', '.bz2', and '.xz'  are assumed to be compressed and transparently
    opened to read like usual files.

    Parameters
    ----------
    filename :  str or bytes
        filename of the file to open or a bytes-stream with the file contents
    mode :  str, optional
        mode in which the file should be opened

    Returns
    -------
    file-handle
        handle of the opened file

    Raises
    ------
    IOError
        If the file does not exist an IOError is raised by the open routine,
        which is not caught within the function
    """
    if config.VERBOSITY >= config.INFO_ALL:
        print(f"XU:io: opening file {filename}")
    if isinstance(filename, bytes):
        fid = io.BytesIO(filename)
    elif filename.endswith('.gz'):
        fid = gzip.open(filename, mode)
    elif filename.endswith('.bz2'):
        fid = bz2.BZ2File(filename, mode)
    elif filename.endswith('.xz'):
        fid = lzma.open(filename, mode)
    else:
        fid = open(filename, mode)

    return fid


class xu_h5open:
    """
    helper object to decide if a HDF5 file has to be opened/closed when
    using with a 'with' statement.
    """

    def __init__(self, f, mode='r'):
        """
        Parameters
        ----------
        f :     str
            filename or h5py.File instance
        mode :  str, optional
            mode in which the file should be opened. ignored in case a file
            handle is passed as f
        """
        self.closeFile = True
        self.fid = None
        self.mode = mode
        if isinstance(f, h5py.File):
            self.fid = f
            self.closeFile = False
            self.filename = f.filename
        elif isinstance(f, str):
            self.filename = f
        else:
            raise InputError("f argument of wrong type was passed, "
                             "should be string or filename")

    def __enter__(self):
        if self.fid:
            if not self.fid.id.valid:
                self.fid = h5py.File(self.filename, self.mode)
        else:
            self.fid = h5py.File(self.filename, self.mode)
        return self.fid

    def __exit__(self, type, value, traceback):
        if self.closeFile:
            self.fid.close()
