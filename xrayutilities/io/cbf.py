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
# Copyright (C) 2013, 2015 Dominik Kriegner <dominik.kriegner@gmail.com>

# module for handling files stored in the CBF data format

import os.path
import re

import numpy

from .. import config, cxrayutilities, utilities
from .filedir import FileDirectory
from .helper import xu_h5open, xu_open

cbf_name_start_num = re.compile(r"^\d")


class CBFFile(object):

    def __init__(self, fname, nxkey="X-Binary-Size-Fastest-Dimension",
                 nykey="X-Binary-Size-Second-Dimension",
                 dtkey="DataType", path=None):
        """
        CBF detector image parser

        Parameters
        ----------
        fname :     str
            name of the CBF file of type .cbf or .cbf.gz
        nxkey :     str, optional
            name of the header key that holds the number of points in
            x-direction
        nykey :     str, optional
            name of the header key that holds the number of points in
            y-direction
        dtkey :     str, optional
            name of the header key that holds the datatype for the binary data
        path :      str, optional
            path to the CBF file
        """

        self.filename = fname
        if path:
            self.full_filename = os.path.join(path, fname)
        else:
            self.full_filename = self.filename

        # evaluate keyword arguments
        self.nxkey = nxkey
        self.nykey = nykey
        self.dtkey = dtkey

        # create attributes for holding data
        self.data = None
        self.ReadData()

    def ReadData(self):
        """
        Read the CCD data into the .data object
        this function is called by the initialization
        """
        with xu_open(self.full_filename, 'rb') as fid:
            tmp = numpy.fromfile(file=fid, dtype="u1").tostring()
            tmp2 = tmp.decode('ascii', 'ignore')
            # read header information
            pos = tmp2.index(self.nxkey + ':') + len(self.nxkey + ':')
            self.xdim = int(tmp2[pos:pos + 6].strip())
            pos = tmp2.index(self.nykey + ':') + len(self.nykey + ':')
            self.ydim = int(tmp2[pos:pos + 6].strip())

            self.data = cxrayutilities.cbfread(tmp, self.xdim, self.ydim)
            self.data.shape = (self.ydim, self.xdim)

    def Save2HDF5(self, h5f, group="/", comp=True):
        """
        Saves the data stored in the EDF file in a HDF5 file as a HDF5 array.
        By default the data is stored in the root group of the HDF5 file - this
        can be changed by passing the name of a target group or a path to the
        target group via the "group" keyword argument.

        Parameters
        ----------
        h5f :   file-handle or str
            a HDF5 file object or name
        group : str, optional
            group where to store the data (default to the root of the file)
        comp :  bool, optional
            activate compression - true by default
        """
        with xu_h5open(h5f, 'a') as h5:
            if isinstance(group, str):
                g = h5.get(group)
            else:
                g = group

            # create the array name
            name = os.path.split(self.filename)[-1]
            name = os.path.splitext(name)[0]
            # perform a second time for case of .cbf.gz files
            name = os.path.splitext(name)[0]
            name = utilities.makeNaturalName(name)
            if cbf_name_start_num.match(name):
                name = "ccd_" + name
            if config.VERBOSITY >= config.INFO_ALL:
                print("xu.io.CBFFile: HDF5 group name: %s" % name)

            # create the array description
            desc = "CBF CCD data from file %s " % (self.filename)

            # create the dataset for the array
            kwds = {'fletcher32': True}
            if comp:
                kwds['compression'] = 'gzip'

            try:
                ca = g.create_dataset(name, data=self.data, **kwds)
            except ValueError:
                del g[name]
                ca = g.create_dataset(name, data=self.data, **kwds)

            ca.attrs['TITLE'] = desc


class CBFDirectory(FileDirectory):

    """
    Parses a directory for CBF files, which can be stored to a HDF5 file for
    further usage
    """

    def __init__(self, datapath, ext="cbf", **keyargs):
        """
        Parameters
        ----------
        datapath :  str
            directory of the CBF files
        ext :       str, optional
            extension of the ccd files in the datapath (default: "cbf")
        keyargs :   dict, optional
            further keyword arguments are passed to CBFFile
        """
        super(CBFDirectory, self).__init__(datapath, ext, CBFFile, **keyargs)
