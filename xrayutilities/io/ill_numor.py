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
# Copyright (C) 2016 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module for reading ILL data files (station D23): numor files
"""

import collections
import os.path
import re

import numpy

from ..exception import InputError
# relative imports from xrayutilities
from .helper import xu_open

re_comment = re.compile(r"^A+$")
re_basicinfo = re.compile(r"^R+$")
re_values = re.compile(r"^F+$")
re_spectrum = re.compile(r"^S+$")
re_header = re.compile(r"^I+$")


class numorFile(object):
    """
    Represents a ILL data file (numor). The file is read during the Constructor
    call. This class should work for created at station D23 using the mad
    acquisition system.

    Parameters
    ----------
    filename :  str
        a string with the name of the data file
    """

    columns = {0: ('detector', 'monitor', 'time', 'gamma', 'omega', 'psi'),
               1: ('detector', 'monitor', 'time', 'gamma'),
               2: ('detector', 'monitor', 'time', 'omega'),
               5: ('detector', 'monitor', 'time', 'psi')}

    def __init__(self, filename, path=None):
        """
        constructor for the data file parser

        Parameters
        ----------
        filename :  str
            a string with the name of the data file
        path :      str, optional
            directory of the data file
        """
        self.filename = filename
        if path is None:
            self.full_filename = self.filename
        else:
            self.full_filename = os.path.join(path, self.filename)

        self.Read()

    def getline(self, fid):
        return fid.readline().decode('ascii')

    def ssplit(self, string):
        """
        multispace split. splits string at two or more spaces after stripping
        it.
        """
        return re.split(r'\s\s+', string.strip())

    def Read(self):
        """
        Read the data from the file
        """
        with xu_open(self.full_filename) as fid:
            self.filesize = os.stat(self.full_filename).st_size
            # read header
            self.init_mopo = {}
            self.comments = []
            self.header = {}
            self._data = []
            while fid.tell() < self.filesize:
                line = self.getline(fid)

                if re_comment.match(line):
                    # read AAAA sections
                    line = self.getline(fid)
                    desc = []
                    for j in range(int(line.split()[1])):
                        desc += self.ssplit(self.getline(fid))
                    comval = self.ssplit(self.getline(fid))
                    self.comments.append((desc, comval))

                if re_basicinfo.match(line):
                    # read RRRR section
                    info = self.ssplit(self.getline(fid))
                    self.dataversion = int(info[2])
                    self.runnumber = int(info[0])

                    if int(info[1]) > 0:
                        headerdesc = ''
                        for j in range(int(info[1])):
                            headerdesc += self.getline(fid) + '\n'
                        self.comments.append((['Fileheader'], [headerdesc]))

                if re_header.match(line):
                    # read IIII section: integer header values
                    info = self.ssplit(self.getline(fid))
                    names = []
                    values = []

                    for j in range(int(info[1])):
                        names += self.getline(fid).split()
                    values = numpy.fromfile(fid, dtype=int,
                                            count=int(info[0]), sep=' ')
                    self.header = {k: v for k, v in zip(names, values)}

                if re_values.match(line):
                    # read FFFF section: initial motor positions
                    info = self.ssplit(self.getline(fid))
                    names = []
                    values = []

                    for j in range(int(info[1])):
                        names += self.ssplit(self.getline(fid))
                    values = numpy.fromfile(fid, dtype=float,
                                            count=int(info[0]), sep=' ')
                    self.init_mopo = {k: v for k, v in zip(names, values)}

                if re_spectrum.match(line):
                    # read SSSS section: initial motor positions
                    info = self.ssplit(self.getline(fid))
                    nspectrum = int(info[0])
                    self.nspectra = int(info[2])

                    if re_values.match(self.getline(fid)):
                        # read FFFF section: subspectrum data
                        nval = int(self.getline(fid))
                        # check if nval is multiple of npdone
                        if nval % self.header['npdone'] != 0:
                            raise InputError("File corrupted, wrong number of "
                                             "data values (%d) found." % nval)

                        self._data.append(numpy.fromfile(fid, dtype=float,
                                                         count=nval, sep=' '))

                    if int(info[1]) == 0:
                        break
            # make data columns accessible by names
            data = numpy.reshape(self._data[0],
                                 (self.header['npdone'],
                                  nval // self.header['npdone']))
            self.data = numpy.rec.fromrecords(
                data, names=self.columns[self.header['manip']])

    def __str__(self):
        ostr = 'Numor: %d (%s)\n' % (self.runnumber, self.filename)
        ostr += 'Comments: %s\n' % " ".join(
            s for c in self.comments for s in c[1])
        ostr += 'Npoints/Ndone: %(nkmes)d/%(npdone)d\n' % (self.header)
        ostr += 'Nspectra: %d\n' % self.nspectra
        ostr += 'Ncolumns: %s' % self.data.shape[1]
        return ostr


def numor_scan(scannumbers, *args, **kwargs):
    """
    function to obtain the angular cooridinates as well as intensity values
    saved in numor datafiles. Especially useful for combining several scans
    into one data object.

    Parameters
    ----------
    scannumbers :   int or str or iterable
        number of the numors, or list of numbers. This will be transformed to a
        string and used as a filename
    args :          str, optional
        names of the motors e.g.: 'omega', 'gamma'
    kwargs :        dict
        keyword arguments are passed on to numorFile. e.g. 'path' for the files
        directory

    Returns
    -------
    [ang1, ang2, ...] :     list
        angular positions list, omitted if no args are given
    data :                  ndarray
        all the data values.

    Examples
    --------
    >>> [om, gam], data = xu.io.numor_scan(414363, 'omega', 'gamma')
    """

    if isinstance(scannumbers, (str, int)):
        scanlist = list([scannumbers])
    elif isinstance(scannumbers, collections.Iterable):
        scanlist = scannumbers
    else:
        raise TypeError('scannumbers is of invalid type (%s)'
                        % type(scannumbers))

    angles = dict.fromkeys(args)
    for key in angles.keys():
        if not isinstance(key, str):
            raise InputError("*arg values need to be strings with motornames")
        angles[key] = numpy.zeros(0)
    buf = numpy.zeros(0)
    MAP = numpy.zeros(0)

    for nr in scanlist:
        scan = numorFile(str(nr), **kwargs)
        sdata = scan.data
        if MAP.dtype == numpy.float64:
            MAP.dtype = sdata.dtype
        # append scan data to MAP, where all data are stored
        MAP = numpy.append(MAP, sdata)
        # check type of scan
        for i in range(len(args)):
            motname = args[i]
            scanlength = len(sdata)
            try:
                buf = sdata[motname]
            except ValueError:
                mv = [v for k, v in scan.init_mopo.items()
                      if motname in k][0]
                buf = mv * numpy.ones(scanlength)
            angles[motname] = numpy.concatenate((angles[motname], buf))

    retval = []
    for motname in args:
        # create return values in correct order
        retval.append(angles[motname])

    if not args:
        return MAP
    elif len(args) == 1:
        return retval[0], MAP
    else:
        return retval, MAP
