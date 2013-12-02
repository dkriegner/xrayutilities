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
class for reading data+header information from tty08 data files

tty08 is system used at beamline P08 at Hasylab Hamburg and creates simple ASCII files to save the data. Information is easily read from the multicolumn data file.
the functions below enable also to parse the information of the header
"""

import re
import numpy
import os

# relative imports from xrayutilities
from .helper import xu_open
from .. import config
from ..exception import InputError

re_columns= re.compile(r"/\*H")
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

    Required constructor arguments:
    ------------------------------
     filename:  a string with the name of the tty08-file

    """

    def __init__(self,filename,path=None):
        self.filename = filename
        if path == None:
            self.full_filename = self.filename
        else:
            self.full_filename = os.path.join(path,self.filename)

        self.Read()

    def Read(self):
        """
        Read the data from the file
        """
    
        with xu_open(self.full_filename) as fid:
            # read header
            self.init_mopo = {}
            while True:
                line = fid.readline()
                #if DEGUG: print line
                if not line:
                    break

                if re_command.match(line):
                    m = line.split(':')
                    self.scan_command = m[1].strip()
                if re_date.match(line):
                    m = line.split(':',1)
                    self.scan_date = m[1].strip()
                if re_epoch.match(line):
                    m = line.split(':',1)
                    self.epoch = float(m[1])
                if re_initmopo.match(line):
                    m = line[3:]
                    m = m.split(';')
                    for e in m:
                        e= e.split('=')
                        self.init_mopo[e[0].strip()] = float(e[1])

                if re_columns.match(line):
                    self.columns = tuple(line.split()[1:])
                    break # here all necessary information is read and we can start reading the data
            self.data = numpy.loadtxt(fid,comments="/")
        
        self.data = numpy.rec.fromrecords(self.data,names=self.columns)

