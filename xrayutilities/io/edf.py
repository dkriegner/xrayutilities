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
# Copyright (C) 2009-2010 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2010-2012,2014-2015
#               Dominik Kriegner <dominik.kriegner@gmail.com>
# Copyright (C) 2012 Tanja Etzelstorfer <tanja.etzelstorfer@jku.at>

# module for handling files stored in the EDF data format developed by the ESRF

import re
import struct
import os.path

import numpy
import h5py

from .helper import xu_open, xu_h5open
from .filedir import FileDirectory
from .. import config

edf_kv_split = re.compile(r"\s*=\s*")  # key value sepeartor for header data
edf_eokv = re.compile(r";")  # end of line for a header
# regular expressions for several ASCII representations of numbers
edf_integer_value = re.compile(r"\d+")
edf_float_value = re.compile(r"[+-]*\d+\.*\d*")
edf_float_e_value = re.compile(r"[+-]*\d+\.\d*e[+-]*\d*")
edf_name_start_num = re.compile(r"^\d")

# dictionary mapping EDF data type keywords onto struct data types
DataTypeDict = {"SignedByte": "b",
                "SignedShort": "h",
                "SignedInteger": "i",
                "SignedLong": "i",
                "FloatValue": "f",
                "DoubleValue": "d",
                "UnsignedByte": "B",
                "UnsignedShort": "H",
                "UnsignedInt": "I",
                "UnsignedLong": "L"}
# SignedLong is only 4byte, on my 64bit machine using SignedLong:"l" caused
# troubles
# UnsignedLong is only 4byte, on my 64bit machine using UnsignedLong:"L"
# caused troubles ("I" works)


def makeNaturalName(name):
    ret = name.replace(" ", "_")
    ret = ret.replace("-", "_")
    ret = ret.replace(".", "_")
    return ret


class EDFFile(object):

    def __init__(self, fname, nxkey="Dim_1", nykey="Dim_2",
                 dtkey="DataType", path="", header=True, keep_open=False):
        """
        required arguments:
        fname ....... name of the EDF file of type .edf or .edf.gz

        keyword arguments:
        nxkey ....... name of the header key that holds the number of
                      points in x-direction
        nykey ....... name of the header key that holds the number of
                      points in y-direction
        dtkey ....... name of the header key that holds the datatype
                      for the binary data
        path ........ path to the EDF file
        header ...... has header (default true)
        keep_open ... if True the file handle is kept open between multiple
                      calls which can cause significant speed-ups
        """

        self.filename = fname
        self.full_filename = os.path.join(path, fname)

        # evaluate keyword arguments
        self.nxkey = nxkey
        self.nykey = nykey
        self.dtkey = dtkey
        self.headerflag = header

        # create attributes for holding data
        self._data = {}
        self._headers = []
        self._data_offsets = []
        self._data_read = False
        self._dimx = []
        self._dimy = []
        self._byte_order = []
        self._fmt_str = []
        self._dtype = []

        self.Parse()
        if keep_open:
            self.fid = xu_open(self.full_filename, 'rb')
        else:
            self.fid = None

        self.nimages = len(self._data_offsets)
        self.header = self._headers[0]

    def Parse(self):
        """
        Parse file to find the number of entries and read the respective
        header information
        """
        header = {}
        offset = 0

        with xu_open(self.full_filename, 'rb') as fid:
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.io.EDFFile.Parse: file: %s" % self.full_filename)

            if self.headerflag:
                while True:  # until end of file
                    hdr_flag = False
                    ml_value_flag = False  # marks a multiline header
                    byte_order = ""
                    for line in fid:  # until end of header
                        line = line.decode('ascii', 'ignore')
                        if config.VERBOSITY >= config.DEBUG:
                            print(line)
                        if line == "":
                            break
                        # remove leading and trailing whitespace symbols
                        line = line.strip()

                        if line == "{" and not hdr_flag:
                            # start with header
                            hdr_flag = True
                            header = {}
                            continue

                        if hdr_flag:
                            # stop reading when the end of the header
                            # is reached
                            if line == "}":
                                # place offset reading here - here we get the
                                # real starting position of the binary data!!
                                offset = fid.tell()
                                break

                            # continue if the line has no content
                            if line == "":
                                continue

                            # split key and value of the header entry
                            if not ml_value_flag:
                                try:
                                    [key, value] = edf_kv_split.split(
                                        line, 1)
                                except:
                                    print("XU.io.EDFFile.Parse: "
                                          "line: %s" % line)

                                key = key.strip()
                                value = value.strip()

                                # if the value extends over multiple lines set
                                # the multiline value flag
                                if value[-1] != ";":
                                    ml_value_flag = True
                                else:
                                    value = value[:-1]
                                    value = value.strip()
                                    header[key] = value
                            else:
                                value = value + line
                                if value[-1] == ";":
                                    ml_value_flag = False

                                    value = value[:-1]
                                    value = value.strip()
                                    header[key] = value
                    else:
                        break
                    # append header to class variables
                    self._byte_order.append(header["ByteOrder"])
                    self._fmt_str.append(DataTypeDict[header[self.dtkey]])
                    self._dimx.append(int(header[self.nxkey]))
                    self._dimy.append(int(header[self.nykey]))
                    self._dtype.append(header[self.dtkey])

                    self._headers.append(header)
                    self._data_offsets.append(offset)
                    # jump over data block
                    tot_nofp = self._dimx[-1] * self._dimy[-1]
                    fid.seek(fid.tell() +
                             struct.calcsize(tot_nofp * self._fmt_str[-1]),
                             0)

            else:  # in case of no header also save one set of defaults
                self._byte_order.append('LowByteFirst')
                self._fmt_str.append(DataTypeDict['UnsignedShort'])
                self._dimx.append(516)
                self._dimy.append(516)
                self._dtype.append('UnsignedShort')
                self._headers.append(header)
                self._data_offsets.append(offset)

        # try to parse motor positions and counters from last found header
        # into separate dictionary
        if 'motor_mne' in header.keys():
            tkeys = header['motor_mne'].split()
            try:
                tval = numpy.array(header['motor_pos'].split(),
                                   dtype=numpy.double)
                self.motors = dict(zip(tkeys, tval))
            except:
                print("XU.io.EDFFile.ReadData: Warning: header conversion "
                      "of motor positions failed")

        if 'counter_mne' in header.keys():
            tkeys = header['counter_mne'].split()
            try:
                tval = numpy.array(header['counter_pos'].split(),
                                   dtype=numpy.double)
                self.counters = dict(zip(tkeys, tval))
            except:
                print("XU.io.EDFFile.ReadData: Warning: header conversion "
                      "of counter values failed")

    def ReadData(self, nimg=0):
        """
        Read the CCD data of the specified image and return the data
        this function is called automatically when the 'data' property is
        accessed, but can also be called manually when only a certain image
        from the file is needed.

        Parameters
        ----------
         nimg:      number of the image which should be read (starts with 0)
        """
        if self.fid:
            binfid = self.fid
            # move to the data section - jump over the header
            binfid.seek(self._data_offsets[nimg], 0)
            # read the data
            tot_nofp = self._dimx[nimg] * self._dimy[nimg]
            fmt_str = self._fmt_str[nimg]
            bindata = binfid.read(struct.calcsize(tot_nofp * fmt_str))
        else:
            with xu_open(self.full_filename, 'rb') as binfid:
                # move to the data section - jump over the header
                binfid.seek(self._data_offsets[nimg], 0)
                # read the data
                tot_nofp = self._dimx[nimg] * self._dimy[nimg]
                fmt_str = self._fmt_str[nimg]
                bindata = binfid.read(struct.calcsize(tot_nofp * fmt_str))
        if config.VERBOSITY >= config.DEBUG:
            print("XU.io.EDFFile: read binary data: nofp: %d len: %d"
                  % (tot_nofp, len(bindata)))
            print("XU.io.EDFFile: format: %s" % fmt_str)

        try:
            num_data = struct.unpack(tot_nofp * fmt_str, bindata)
        except:
            if fmt_str == 'L':
                fmt_str = 'I'
                try:
                    num_data = struct.unpack(tot_nofp * fmt_str, bindata)
                except:
                    raise IOError("XU.io.EDFFile: data format (%s) has "
                                  "different byte-length, from amount of data "
                                  "one expects %d bytes per entry"
                                  % (fmt_str, len(bindata) / tot_nofp))
            else:
                raise IOError("XU.io.EDFFile: data format (%s) has different "
                              "byte-length, from amount of data one expects "
                              "%d bytes per entry"
                              % (fmt_str, len(bindata) / tot_nofp))

        # find the proper datatype
        if self._dtype[nimg] == "SignedByte":
            data = numpy.asarray(num_data, dtype=numpy.int8)
        elif self._dtype[nimg] == "SignedShort":
            data = numpy.asarray(num_data, dtype=numpy.int16)
        elif self._dtype[nimg] == "SignedInteger":
            data = numpy.asarray(num_data, dtype=numpy.int32)
        elif self._dtype[nimg] == "SignedLong":
            data = numpy.asarray(num_data, dtype=numpy.int64)
        elif self._dtype[nimg] == "FloatValue":
            data = numpy.asarray(num_data, dtype=numpy.float)
        elif self._dtype[nimg] == "DoubleValue":
            data = numpy.asarray(num_data, dtype=numpy.double)
        elif self._dtype[nimg] == "UnsignedByte":
            data = numpy.asarray(num_data, dtype=numpy.uint8)
        elif self._dtype[nimg] == "UnsignedShort":
            data = numpy.asarray(num_data, dtype=numpy.uint16)
        elif self._dtype[nimg] == "UnsignedInt":
            data = numpy.asarray(num_data, dtype=numpy.uint32)
        elif self._dtype[nimg] == "UnsignedLong":
            data = numpy.asarray(num_data, dtype=numpy.uint64)
        else:
            data = numpy.asarray(num_data, dtype=numpy.double)

        data.shape = (self._dimy[nimg], self._dimx[nimg])

        if self._byte_order[nimg] != "LowByteFirst":  # data = data.byteswap()
            print("XU.io.EDFFile.ReadData: check byte order - "
                  "not low byte first")

        return data

    @property
    def data(self):
        if not self._data_read:
            for i in range(self.nimages):
                self._data[i] = self.ReadData(i)
            self._data_read = True
        if self.nimages == 1:
            return self._data[0]
        else:
            return self._data

    def Save2HDF5(self, h5f, group="/", comp=True):
        """
        Saves the data stored in the EDF file in a HDF5 file as a HDF5 array.
        By default the data is stored in the root group of the HDF5 file - this
        can be changed by passing the name of a target group or a path to the
        target group via the "group" keyword argument.

        Parameters
        ----------
         h5f ...... a HDF5 file object or name

        optional keyword arguments:
         group .... group where to store the data (default to the root of the
                    file)
         comp ..... activate compression - true by default
        """
        with xu_h5open(h5f, 'a') as h5:
            if isinstance(group, str):
                if group == '/':
                    g = h5
                else:
                    if group in h5:
                        del h5[group]
                    g = h5.create_group(group)
            else:
                g = group

            # create the array name
            ca_name = os.path.split(self.filename)[-1]
            ca_name = os.path.splitext(ca_name)[0]
            # perform a second time for case of .edf.gz files
            ca_name = os.path.splitext(ca_name)[0]
            ca_name = makeNaturalName(ca_name)
            if edf_name_start_num.match(ca_name):
                ca_name = "ccd_" + ca_name
            if config.VERBOSITY >= config.INFO_ALL:
                print(ca_name)

            # create the array description
            ca_desc = "EDF CCD data from file %s " % (self.filename)
            kwds = {'fletcher32': True}
            if comp:
                kwds['compression'] = 'gzip'

            if self.nimages != 1:
                ca_name += '_{n:04d}'

            d = self.data
            for n in range(self.nimages):
                if self.nimages != 1:
                    d = self.data[n]
                name = ca_name.format(n=n)
                try:
                    ca = g.create_dataset(name, data=d, **kwds)
                except ValueError:
                    del g[name]
                    ca = g.create_dataset(name, data=d, **kwds)

                ca.attrs['TITLE'] = ca_desc

                # finally we have to append the attributes
                for k in self.header.keys():
                    ca.attrs[makeNaturalName(k)] = self.header[k]


class EDFDirectory(FileDirectory):

    """
    Parses a directory for EDF files, which can be stored to a HDF5 file for
    further usage
    """

    def __init__(self, datapath, ext="edf", **keyargs):
        """

        required arguments:
        datapath ..... directory of the EDF file

        optional keyword arguments:
        ext .......... extension of the ccd files in the datapath
                       (default: "edf")

        further keyword arguments are passed to EDFFile
        """
        super(EDFDirectory, self).__init__(datapath, ext, EDFFile, **keyargs)
