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
# Copyright (C) 2013-2021 Dominik Kriegner <dominik.kriegner@gmail.com>


"""
class for reading data + header information from tty08 data files

tty08 is a system used at beamline P08 at Hasylab Hamburg and creates simple
ASCII files to save the data. Information is easily read from the multicolumn
data file. the functions below enable also to parse the information of the
header
"""

import glob
import os.path
import re

import numpy
import numpy.lib.recfunctions

from ..exception import InputError
# relative imports from xrayutilities
from .helper import xu_open

re_columns = re.compile(r"/\*H")
re_command = re.compile(r"^/\*C command")
re_comment = re.compile(r"^/\*")
re_date = re.compile(r"^/\*D date")
re_epoch = re.compile(r"^/\*T epoch")
re_initmopo = re.compile(r"^/\*M")


class tty08File(object):

    """
    Represents a tty08 data file. The file is read during the
    Constructor call. This class should work for data stored at
    beamline P08 using the tty08 acquisition system.

    Parameters
    ----------
    filename :  str
        tty08-filename
    mcadir :    str, optional
        directory name of MCA files
    """

    def __init__(self, filename, path=None, mcadir=None):
        self.filename = filename
        if path is None:
            self.full_filename = self.filename
        else:
            self.full_filename = os.path.join(path, self.filename)

        self.Read()

        if mcadir is not None:
            self.mca_directory = mcadir
            self.mca_files = sorted(glob.glob(
                os.path.join(self.mca_directory, '*')))

            if self.mca_files:
                self.ReadMCA()

    def ReadMCA(self):
        self.mca = numpy.empty((len(self.mca_files),
                                numpy.loadtxt(self.mca_files[0]).shape[0]),
                               dtype=float)
        for i in range(len(self.mca_files)):
            mcadata = numpy.loadtxt(self.mca_files[i])

            self.mca[i, :] = mcadata[:, 1]

            if i == 0:
                if len(mcadata.shape) == 2:
                    self.mca_channels = mcadata[:, 0]
                else:
                    self.mca_channels = numpy.arange(0, mcadata.shape[0])

        mcatemp = self.mca.view([('MCA',
                                  (self.mca.dtype, self.mca.shape[1]))])
        self.data = numpy.lib.recfunctions.merge_arrays([self.data, mcatemp],
                                                        flatten=True)

    def Read(self):
        """
        Read the data from the file
        """

        with xu_open(self.full_filename) as fid:
            # read header
            self.init_mopo = {}
            for line in fid:
                line = line.decode('ascii')

                if re_command.match(line):
                    m = line.split(':')
                    self.scan_command = m[1].strip()
                if re_date.match(line):
                    m = line.split(':', 1)
                    self.scan_date = m[1].strip()
                if re_epoch.match(line):
                    m = line.split(':', 1)
                    self.epoch = float(m[1])
                if re_initmopo.match(line):
                    m = line[3:]
                    m = m.split(';')
                    for e in m:
                        e = e.split('=')
                        self.init_mopo[e[0].strip()] = float(e[1])

                if re_columns.match(line):
                    self.columns = tuple(line.split()[1:])
                    # here all necessary information is read and we can start
                    # reading the data
                    break
            self.data = numpy.loadtxt(fid, comments="/")

        self.data = numpy.rec.fromrecords(self.data, names=self.columns)


def gettty08_scan(scanname, scannumbers, *args, **keyargs):
    """
    function to obtain the angular cooridinates as well as intensity values
    saved in TTY08 datafiles. Especially usefull for reciprocal space map
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

            - `omname`: the name of the omega motor (or its equivalent)
            - `ttname`: the name of the two theta motor (or its equivalent)

    keyargs :       dict, optional
        keyword arguments are passed on to tty08File

    Returns
    -------
    [ang1, ang2, ...] :     list, optional
        angular positions of the center channel of the position sensitive
        detector (numpy.ndarray 1D), omitted if no `args` are given
    MAP :                   ndarray
        All the data values as stored in the data file (includes the
        intensities e.g. MAP['MCA']).

    Examples
    --------
    >>> [om, tt], MAP = xu.io.gettty08_scan('text%05d.dat', 36, 'omega',
    >>>                                     'gamma')
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
        scan = tty08File(scanname % nr, **keyargs)
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
    elif len(args) == 1:
        return retval[0], MAP
    else:
        return retval, MAP
