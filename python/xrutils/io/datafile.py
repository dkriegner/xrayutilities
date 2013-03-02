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
# Copyright (C) 2010 Dominik Kriegner <dominik.kriegner@gmail.com>

#an experimental data file
import os.path

class DataFile(object):
    """
    class DataFile
    A data file object holds
    basic information about a data file that is somewhere
    stored on the file system. However, it knows nothing
    about the real internals of the file (format and so on).
    It stores only information like the path, filename,
    and certain offsets within the file.
    """

    def __init__(self,path,filename):
        self.path = path

        self.filename = filename
        self.fullfilename = os.path.join(self.path,self.filename)
        self.fid = None

    def open(self):
        """
        Open the file in the file object.
        """
        try:
            self.fid = open(self.fullfilename,"r")
        except:
            self.fid = None
            raise IOError("error opening file %s" %self.fullfilename)

        #after opening the file we have to find EOF offset with
        #respect to the beginning of the file

    def close(self):
        try:
            self.fid.close()
        except:
            self.fid = None
            return None


