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
# Copyright (c) 2015-2019, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>


"""
class for reading data + header information from Rigaku RAS (3-column ASCII)
files

Such datafiles are generated by the Smartlab Guidance software from Rigaku.
"""

import os.path
import re
from itertools import islice

import numpy
import numpy.lib.recfunctions

from .. import config
from ..exception import InputError
# relative imports from xrayutilities
from .helper import xu_open

re_measstart = re.compile(r"^\*RAS_DATA_START")
re_measend = re.compile(r"^\*RAS_DATA_END")
re_headerstart = re.compile(r"^\*RAS_HEADER_START")
re_headerend = re.compile(r"^\*RAS_HEADER_END")
re_datastart = re.compile(r"^\*RAS_INT_START")
re_dataend = re.compile(r"^\*RAS_INT_END")
re_scanaxis = re.compile(r"^\*MEAS_SCAN_AXIS_X_INTERNAL")
re_intstart = re.compile(r"^\*RAS_INT_START")
re_datestart = re.compile(r"^\*MEAS_SCAN_START_TIME")
re_datestop = re.compile(r"^\*MEAS_SCAN_END_TIME")
re_initmoponame = re.compile(r"^\*MEAS_COND_AXIS_NAME_INTERNAL")
re_initmopovalue = re.compile(r"^\*MEAS_COND_AXIS_POSITION")
re_datacount = re.compile(r"^\*MEAS_DATA_COUNT")
re_measspeed = re.compile(r"^\*MEAS_SCAN_SPEED ")
re_measstep = re.compile(r"^\*MEAS_SCAN_STEP ")


class RASFile:

    """
    Represents a RAS data file. The file is read during the
    constructor call

    Parameters
    ----------
    filename :  str
        name of the ras-file
    path :      str, optional
        path to the data file
    """

    def __init__(self, filename, path=None):
        self.filename = filename
        if path is None:
            self.full_filename = self.filename
        else:
            self.full_filename = os.path.join(path, self.filename)

        self.scans = []
        self.Read()

    def Read(self):
        """
        Read the data from the file
        """
        with xu_open(self.full_filename) as fid:
            while True:
                t = fid.tell()
                line = fid.readline()
                line = line.decode('ascii', 'ignore')
                if config.VERBOSITY >= config.DEBUG:
                    print(f"XU.io.RASFile: {t}: '{line}'")
                if re_measstart.match(line):
                    continue
                if re_headerstart.match(line):
                    s = RASScan(self.full_filename, t)
                    self.scans.append(s)
                    fid.seek(s.fidend)  # set handle to after scan
                elif re_measend.match(line) or line in (None, ''):
                    break
                else:
                    continue
        if len(self.scans) > 0:
            self.scan = self.scans[0]


class RASScan:

    """
    Represents a single Scan portion of a RAS data file. The scan is parsed
    during the constructor call

    Parameters
    ----------
    filename :  str
        file name of the data file
    pos :       int
        seek position of the 'RAS_HEADER_START' line
    """

    def __init__(self, filename, pos):
        self.filename = filename
        self.fidpos = pos
        self.fidend = pos
        with xu_open(self.filename) as self.fid:
            self.fid.seek(self.fidpos)
            self._parse_header()
            self._parse_data()
            self.fidend = self.fid.tell()

    def _parse_header(self):
        """
        Read the data from the file
        """
        # read header
        self.header = []
        keys = {}
        position = {}
        offset = self.fid.tell()
        for line in self.fid:
            offset += len(line)
            line = line.decode('ascii', 'ignore')
            self.header.append(line)
            if config.VERBOSITY >= config.DEBUG:
                print("XU.io.RASScan: %d: '%s'" % (offset, line))

            if re_datestart.match(line):
                m = line.split(' ', 1)[-1].strip()
                self.scan_start = m.strip('"')
            elif re_datestop.match(line):
                m = line.split(' ', 1)[-1].strip()
                self.scan_stop = m.strip('"')
            elif re_initmoponame.match(line):
                idx = int(line.split('-', 1)[-1].split()[0])
                moname = line.split(' ', 1)[-1].strip().strip('"')
                keys[idx] = moname
            elif re_initmopovalue.match(line):
                idx = int(line.split('-', 1)[-1].split()[0])
                mopos = line.split(' ', 1)[-1].strip().strip('"')
                try:
                    mopos = float(mopos)
                except ValueError:
                    pass
                position[idx] = mopos
            elif re_scanaxis.match(line):
                self.scan_axis = line.split(' ', 1)[-1].strip().strip('"')
            elif re_datacount.match(line):
                length = line.split(' ', 1)[-1].strip().strip('"')
                self.length = int(float(length))
            elif re_measspeed.match(line):
                speed = line.split(' ', 1)[-1].strip().strip('"')
                self.meas_speed = float(speed)
            elif re_measstep.match(line):
                step = line.split(' ', 1)[-1].strip().strip('"')
                self.meas_step = float(step)
            elif re_headerend.match(line):
                break

        # generate header dictionary
        self.init_mopo = {}
        for k in keys:
            self.init_mopo[keys[k]] = position[k]
        self.fid.seek(offset)

    def _parse_data(self):
        line = self.fid.readline().decode('ascii', 'ignore')
        offset = self.fid.tell()
        if re_datastart.match(line):
            lines = islice(self.fid, self.length)
            self.data = numpy.genfromtxt(lines)
            self.data = numpy.rec.fromrecords(self.data,
                                              names=[self.scan_axis,
                                                     'int',
                                                     'att'])
            self.fid.seek(offset)
            lines = islice(self.fid, self.length)
            dlength = numpy.sum([len(line) for line in lines])
            if config.VERBOSITY >= config.DEBUG:
                print("XU.io.RASScan: offset %d; data-length %d"
                      % (offset, dlength))
            self.fid.seek(offset + dlength)
        else:
            raise IOError('File handle at wrong position to read data!')


def getras_scan(scanname, scannumbers, *args, **kwargs):
    """
    function to obtain the angular cooridinates as well as intensity values
    saved in RAS datafiles. Especially useful for reciprocal space map
    measurements, and to combine date from several scans

    further more it is possible to obtain even more positions from
    the data file if more than two string arguments with its names are given

    Parameters
    ----------
    scanname :      str
        name of the scans, for multiple scans this needs to be a template
        string
    scannumbers :   int, tuple or list
        number of the scans of the reciprocal space map
    args :          str, optional
        names of the motors. to read reciprocal space maps measured in coplanar
        diffraction give:

            - omname:  name of the omega motor (or its equivalent)
            - ttname:  name of the two theta motor (or its equivalent)

    kwargs :        dict
        keyword arguments forwarded to RASFile function

    Returns
    -------
    [ang1, ang2, ...] :     list
        angular positions are extracted from the respective scan header, or
        motor positions during the scan. this is omitted if no `args` are given
    rasdata :   ndarray
        the data values (includes the intensities e.g. rasdata['int']).

    Examples
    --------
    >>> [om, tt], MAP = xu.io.getras_scan('text%05d.ras', 36, 'Omega',
    >>>                                   'TwoTheta')
    """

    if isinstance(scannumbers, (list, tuple)):
        scanlist = scannumbers
    else:
        scanlist = list([scannumbers])

    angles = dict.fromkeys(args)
    for key in angles:
        if not isinstance(key, str):
            raise InputError("*arg values need to be strings with motornames")
        angles[key] = numpy.zeros(0)
    buf = numpy.zeros(0)
    MAP = numpy.zeros(0)

    for nr in scanlist:
        rasfile = RASFile(scanname % nr, **kwargs)
        for scan in rasfile.scans:
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
                    buf = scan.init_mopo[motname] * numpy.ones(scanlength)
                angles[motname] = numpy.concatenate((angles[motname], buf))

    retval = []
    for motname in args:
        # create return values in correct order
        retval.append(angles[motname])

    if not args:
        return MAP
    if len(args) == 1:
        return retval[0], MAP
    return retval, MAP
