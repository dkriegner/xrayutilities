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
# Copyright (C) 2009 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2009-2010,2013 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
a set of  routines to convert Seifert ASCII files to HDF5
in fact there exist two posibilities how the data is stored (depending on the
use detector):

 1. as a simple line scan (using the point detector)
 2. as a map using the PSD

In the first case the data ist stored
"""

import re
import os.path
import itertools

import numpy

from .helper import xu_open
from .. import config

# define some regular expressions
nscans_re = re.compile(r"^&NumScans=\d+")
scan_data_re = re.compile(r"^\#Values\d+")
scan_partab_re = re.compile(r"^\#ScanTableParameter")
novalues_re = re.compile(r"^&NoValues=\d+")
scanaxis_re = re.compile(r"^&ScanAxis=.*")

# some constant regular expressions
re_measparams = re.compile(r"#MeasParameter")
re_rsmparams = re.compile(r"#RsmParameter")
re_data = re.compile(r"#Values")
re_keyvalue = re.compile(r"&\S+=\S")
re_invalidkeyword = re.compile(r"^\d+\S")
re_multiblank = re.compile(r"\s+")
re_position = re.compile(r"&Pos=[+-]*\d*\.\d*")
re_start = re.compile(r"&Start=[+-]*\d*\.\d*")
re_end = re.compile(r"&End=[+-]*\d*\.\d*")
re_step = re.compile(r"&Step=[+-]*\d*\.\d*")
re_time = re.compile(r"&Time=\d*.\d*")
re_stepscan = re.compile(r"^&Start")
re_dataline = re.compile(r"^[+-]*\d*\.\d*")
re_absorber = re.compile(r"^&Axis=A6")


def repair_key(key):
    """
    Repair a key string in the sense that the string is changed in a way that
    it can be used as a valid Python identifier. For that purpose all blanks
    within the string will be replaced by _ and leading numbers get an
    preceeding _.
    """

    if re_invalidkeyword.match(key):
        key = "_" + key

    # now replace all blanks
    key = key.replace(" ", "_")

    return key


class SeifertHeader(object):
    """
    helper class to represent a Seifert (NJA) scan file header
    """
    def __init__(self):
        pass

    def __str__(self):
        ostr = ""
        for k in self.__dict__.keys():
            value = self.__getattribute__(k)
            if isinstance(value, float):
                ostr += k + " = %f\n" % value
            else:
                ostr += k + " = %s\n" % value

        return ostr


class SeifertMultiScan(object):
    """
    Class to parse a Seifert (NJA) multiscan file
    """
    def __init__(self, filename, m_scan, m2, path=None):
        """
        Parse data from a multiscan Seifert file.

        required input arguments:
         filename ................... name of the NJA file
         m_scan ..................... name of the scan axis
         m2 ......................... name of the second moving motor
         path ....................... common path to datafile
        """
        if path:
            self.Filename = os.path.join(path, filename)
        else:
            self.Filename = filename

        try:
            self.fid = xu_open(self.Filename)
        except:
            self.fid = None
            raise IOError(
                "error opening Seifert datafile %s" %
                (self.Filename))

        self.nscans = 0  # total number of scans
        self.npscan = 0  # number of points per scan
        self.ctime = 0  # counting time
        self.re_m2 = re.compile(r"^&Axis=%s\s+&Task=Drive" % m2)
        self.re_sm = re.compile(r"^&ScanAxis=%s" % m_scan)
        self.scan_motor_name = m_scan
        self.sec_motor_name = m2

        self.m2_pos = []
        self.sm_pos = []
        self.data = []
        self.n_sm_pos = 0

        if self.fid:
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.io.SeifertScan: parsing file: %s" % self.Filename)
            self.parse()

    def parse(self):
        self.data = []
        m2_tmppos = None
        self.sm_pos = []
        self.m2_pos = []

        # flag to check if all header information was parsed
        header_complete = False

        while True:
            lb = list(itertools.islice(self.fid, 1))
            if not lb:
                break
            lb = lb[0].decode('ascii').strip()

            # the first thing needed is the number of scans in the file (in
            # file header)
            if nscans_re.match(lb):
                t = lb.split("=")[1]
                self.nscans = int(t)

            if self.re_m2.match(lb):
                t = re_position.findall(lb)[0]
                t = t.split("=")[1]
                m2_tmppos = float(t)

            if novalues_re.match(lb):
                t = lb.split("=")[1]
                self.n_sm_pos = int(t)
                header_complete = True

            if header_complete:
                # append motor positions of second motor
                self.m2_pos.append([[m2_tmppos] * self.n_sm_pos])

                # reset header flag
                header_complete = False
                # read data lines (number of lines determined by number of
                # values)
                datalines = itertools.islice(self.fid, self.n_sm_pos)
                t = numpy.loadtxt(datalines)
                self.data.append(t[:, 1])
                self.sm_pos.append(t[:, 0])

        # after reading all the data
        self.m2_pos = numpy.array(self.m2_pos, dtype=numpy.double)
        self.sm_pos = numpy.array(self.sm_pos, dtype=numpy.double)
        self.data = numpy.array(self.data, dtype=numpy.double)

        self.data.shape = (self.nscans, self.n_sm_pos)
        self.m2_pos.shape = (self.nscans, self.n_sm_pos)
        self.sm_pos.shape = (self.nscans, self.n_sm_pos)


class SeifertScan(object):
    """
    Class to parse a single Seifert (NJA) scan file
    """
    def __init__(self, filename, path=None):
        """
        Constructor for a SeifertScan object.

        required input arguments:
         filename:  a string with the name of the file to read
         path:      common path to the filenames

        """
        if path:
            self.Filename = os.path.join(path, filename)
        else:
            self.Filename = filename

        try:
            self.fid = xu_open(self.Filename)
        except:
            self.fid = None
            raise IOError(
                "error opening Seifert datafile %s" %
                (self.Filename))

        self.hdr = SeifertHeader()
        self.data = []
        self.axispos = {}

        if self.fid:
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.io.SeifertScan: parsing file: %s" % self.Filename)
            self.parse()

        try:
            if self.hdr.NumScans != 1:
                self.data.shape = (int(self.data.shape[0] / self.hdr.NoValues),
                                   int(self.hdr.NoValues), 2)
        except:
            pass

    def parse(self):
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.io.SeifertScan.parse: starting the parser")
        self.data = []
        while True:
            lb = self.fid.readline().decode('ascii')
            if not lb:
                break
            # remove leading and trailing whitespace and newline characeters
            lb = lb.strip()

            # every line is broken into its content
            llist = re_multiblank.split(lb)
            tmplist = []
            axes = ""
            for e in llist:
                # if the entry is a key value pair
                if re_keyvalue.match(e):
                    (key, value) = e.split("=")
                    # remove leading & from the key
                    key = key[1:]
                    # have to manage malformed key names that cannot be used as
                    # Python identifiers (leading numbers or blanks inside the
                    # name)
                    key = repair_key(key)

                    # try to convert the values to float numbers
                    # leave them as strings if this is not possible
                    try:
                        value = float(value)
                    except:
                        pass

                    if key == "Axis":
                        axes = value
                        try:
                            self.axispos[value]
                        except:
                            self.axispos[value] = []
                    elif key == "Pos":
                        self.axispos[axes] += [value, ]

                    self.hdr.__setattr__(key, value)
                else:
                    try:
                        tmplist.append(float(e))
                    except:
                        pass

            if tmplist != []:
                self.data.append(tmplist)

        # in the end we convert the data list to a numeric array
        self.data = numpy.array(self.data, dtype=numpy.float)
        for key in self.axispos:
            self.axispos[key] = numpy.array(self.axispos[key])


def getSeifert_map(filetemplate, scannrs=None, path=".", scantype="map",
                   Nchannels=1280):
    """
    parses multiple Seifert *.nja files and concatenates the results.  for
    parsing the xrayutilities.io.SeifertMultiScan class is used. The function
    can be used for parsing maps measured with the Meteor1D and point detector.

    Parameter
    ---------
     filetemplate:  template string for the file names, can contain
                    a %d which is replaced by the scan number or be a
                    list of filenames
     scannrs:       int or list of scan numbers
     path:          common path to the filenames
     scantype:      type of datafile: can be either "map" (reciprocal space map
                    measured with a regular Seifert job (default)) or "tsk"
                    (MCA spectra measured using the TaskInterpreter)

     Nchannels:     number of channels of the MCA (needed for "tsk"
                    measurements)

    Returns
    -------
     om,tt,psd: as flattened numpy arrays

    Example
    -------
     >>> om,tt,psd = xrayutilities.io.getSeifert_map("samplename_%d.xrdml",
                                                     [1,2], path="./data")
    """
    # read raw data and convert to reciprocal space
    om = numpy.zeros(0)
    tt = numpy.zeros(0)
    if scantype == "map":
        psd = numpy.zeros(0)
    else:
        psd = numpy.zeros((0, Nchannels))
    # create scan names
    if scannrs is None:
        files = [filetemplate]
    else:
        files = list()
        if not getattr(scannrs, '__iter__', False):
            scannrs = [scannrs]
        for nr in scannrs:
            files.append(filetemplate % nr)

    # parse files
    for f in files:
        if scantype == "map":
            d = SeifertMultiScan(os.path.join(path, f), 'T', 'O')

            om = numpy.concatenate((om, d.m2_pos.flatten()))
            tt = numpy.concatenate((tt, d.sm_pos.flatten()))
            psd = numpy.concatenate((psd, d.data.flatten()))
        else:  # scantype == "tsk":
            d = SeifertScan(os.path.join(path, f))

            om = numpy.concatenate((om, d.axispos['O'].flatten()))
            tt = numpy.concatenate((tt, d.axispos['T'].flatten()))
            psd = numpy.concatenate((psd, d.data[:, :, 1]))

    return om, tt, psd
