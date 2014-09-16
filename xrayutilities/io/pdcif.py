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
# Copyright (C) 2014 Dominik Kriegner <dominik.kriegner@gmail.com>

import numpy
import re
import copy
import shlex

from . import xu_open

re_label = re.compile(r'^\s*_')
re_default = re.compile(r'^\s*_(pd_meas_counts_total|pd_meas_intensity_total|pd_proc_intensity_total|pd_proc_intensity_net|pd_calc_intensity_total|pd_calc_intensity_net)')
re_loop = re.compile(r'^\s*loop_')
re_nop = re.compile(r'^\s*_(pd_meas_number_of_points|pd_meas_detector_id)')
re_multiline = re.compile(r';')

class pdCIF(object):
    """
    the class implements a primitive parser for pdCIF-like files.  It reads
    every entry and collects the information in the header attribute. The first
    loop containing one of the intensity fields is assumed to be the data the
    user is interested in and is transfered to the data array which is stored
    as numpy record array the columns can be accessed by name

    intensity fields:
      _pd_meas_counts_total, _pd_meas_intensity_total,
      _pd_proc_intensity_total, _pd_proc_intensity_net,
      _pd_calc_intensity_total, _pd_calc_intensity_net

    alternatively the data column name can be given as argument to the constructor
    """
    def __init__(self,filename,datacolumn=None):
        """
        contructor of the pdCIF class

        Parameters
        ----------
         filename:      filename of the file to be parsed
         datacolumn:    name of data column to identify the data loop
                        (default =None; means that a list of default names is used)
        """
        self.filename = filename
        self.datacolumn = datacolumn
        self.header = {}
        self.data = None

        self.Parse()

    def Parse(self):
        """
        parser of the pdCIF file. the method reads the data from the file and
        fills the data and header attributes with content
        """

        with xu_open(self.filename) as fh:
            self._parse_single(fh)

    def _parse_single(self,fh):
        """
        internal routine to parse a single loop of the pdCIF file

        Parameters
        ----------
         fh:    file handle
        """
        loopStart = False
        dataLoop = False
        loopheader = []
        numOfEntries = -1
        multiline = None

        while True:
            line = fh.readline().decode('ascii')
            if not line:
                break

            if re_loop.match(line):
                loopStart = True
                continue

            if multiline:
                multiline += line
                if re_multiline.match(line): # end of multiline
                    val = multiline
                    self.header[label] = val
                    multiline = None
                continue

            if re_label.match(line) and not loopStart:
                # parse header
                split = line.split(None,1)
                label = split[0].strip()
                try:
                    val = split[1].strip()
                    self.header[label] = val
                    # convert data format of header line
                    if re_nop.match(line):
                        numOfEntries = int(val)
                    try: self.header[label] = float(val)
                    except ValueError: self.header[label] = val
                except IndexError:
                    # try if multiline
                    line2 = fh.readline().decode('ascii')
                    if re_multiline.match(line2):
                        multiline = line2
                    else:
                        fh.seek(fh.tell()-len(line2))
                        raise ValueError('a value is missing for label %s'%label)

            elif re_label.match(line) and loopStart:
                # read loop entries
                if (self.datacolumn == None and re_default.match(line)) or line.strip() == self.datacolumn:
                    dataLoop = True
                loopheader.append(line.strip())

            elif loopStart:
                fh.seek(fh.tell()-len(line))
                if numOfEntries != -1 and dataLoop:
                    self.data = self._parse_loop_numpy(fh,loopheader,numOfEntries)
                    break
                elif dataLoop:
                    self._parse_loop(fh,loopheader)
                    length = len(self.header[loopheader[0]])
                    dtypes = [ (entry,type(self.header[entry][0])) for entry in loopheader ]
                    for i in range(len(dtypes)):
                        if dtypes[i][1] == str: dtypes[i] = (dtypes[i][0],numpy.str_,64)
                    self.data = numpy.zeros(length,dtype = dtypes)
                    for entry in loopheader: self.data[entry] = self.header.pop(entry)
                else:
                    self._parse_loop(fh,loopheader)
                dataLoop = False
                loopStart = False
                loopheader = []
                numOfEntries = -1

    def _parse_loop_numpy(self,filehandle,fields,nentry):
        """
        function to parse a loop using numpy routines

        Parameter
        ---------
         filehandle:    filehandle object to use as data source
         fields:        field names in the loop
         nentry:        number of entries in the loop

        Return
        ------
         data:          data read from the file as numpy record array
        """
        tmp = numpy.fromfile(filehandle,count=nentry*len(fields),sep=' ')
        data = numpy.rec.fromarrays(tmp.reshape((-1,len(fields))).T,names=fields)
        return data

    def _parse_loop(self,filehandle,fields):
        """
        function to parse a loop using python loops routines. the fields are added
        to the fileheader dictionary

        Parameter
        ---------
         filehandle:    filehandle object to use as data source
         fields:        field names in the loop

        Return
        ------
         nothing
        """
        fh = filehandle

        for f in fields:
            self.header[f] = []
        while True:
            line = fh.readline().decode('ascii')
            if not line:
                break

            if re_label.match(line) or line.strip() == '':
                fh.seek(fh.tell()-len(line))
                break
            row = shlex.split(line)
            for i in range(len(fields)):
                try: self.header[fields[i]].append(float(row[i]))
                except ValueError: self.header[fields[i]].append(row[i])
                except IndexError: # maybe multiline field
                    line2 = fh.readline().decode('ascii')
                    if re_multiline.match(line2):
                        multiline = line2
                        while True:
                            line = fh.readline().decode('ascii')
                            if not line:
                                break
                            if not re_multiline.match(line):
                               multiline += line
                            else:
                                self.header[fields[i]].append(multiline)
                                break
                    else:
                        fh.seek(fh.tell()-len(line2))
                        raise ValueError('a column is missing for label %s in a loop'%fields[i])


class pdESG(pdCIF):
    """
    class for parsing multiple pdCIF loops in one file.
    This includes especially *.esg files which are supposed to
    consist of multiple loops of pdCIF data with equal length.

    Upon parsing the class tries to combine the data of these different
    scans into a single data matrix -> same shape of subscan data is assumed
    """
    def __init__(self,filename,datacolumn=None):
        self.filename = filename
        self.datacolumn = datacolumn
        self.fileheader = {}
        self.header = {}
        self.data = None

        self.Parse()

    def Parse(self):
        """
        parser of the pdCIF file. the method reads the data from the file and
        fills the data and header attributes with content
        """

        with xu_open(self.filename) as fh:
            # parse first header and loop
            self._parse_single(fh)
            self.fileheader = copy.deepcopy(self.header)
            self.header = {}
            fdata = self.data
            datasize = self.data.size
            nscan = 1
            tell = 0
            while True: # try to parse all scans
                tell = fh.tell()
                try:
                    self._parse_single(fh)
                except:
                    break
                if tell == fh.tell(): break
                # copy changing data from header
                for key in self.header:
                    if key in self.fileheader:
                        if not isinstance(self.fileheader[key],list):
                            self.fileheader[key] = [self.fileheader[key],]
                        self.fileheader[key].append(self.header[key])
                    else:
                        self.fileheader[key] = self.header[key]

                fdata = numpy.append(fdata,self.data)
                nscan +=1

        # convert data for output to user
        for key in self.fileheader:
            if isinstance(self.fileheader[key],list):
                self.fileheader[key] = numpy.array(self.fileheader[key])
        self.data = numpy.empty(fdata.shape)
        self.data[...] = fdata[...]
        self.data.shape = (nscan,datasize)

