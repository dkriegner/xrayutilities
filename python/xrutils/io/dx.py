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

#a module to write OpenDX input files in an object oriented approach
import os.path

class DXFile(object):
    """
    Class DXFile:
        This class is the basement for the dx module. It represents a
        complete DX file and acts as a container for all other
        objects residing within this files. I.g. this are fields.

    Required input arguments for the constructor:
        filename ............... name of the DX file

    Optional keyword arguments:
        seq .................... yes or no, this flag determines
                                 wether a file contains a sequence or not.
        path ................... path where to store the file.
    """

    def __init__(self,filename,**keyargs):

        #parse the keyword arguments:
        if "seq" in keyargs:
            #file contains a sequence
            self.seq_flag = 1
        else:
            self.seq_flag = 0

        if "path" in keyargs:
            self.DXFilePath = keyargs["path"]
        else:
            self.DXFilePath = "."

        self.DXFileName = filename

        #try to open the file
        try:
            self.DXFile_FID = open(os.path.join(self.DXFilePath,self.DXFileName))
        except:
            raise IOError("error opening file: %s" %os.path.join(self.DXFilePath,self.DXFileName))



    def __str__(self):
        pass

class DXGrid(object):

    def __init__(self,**keyargs):
        pass

class DXConnections(object):

    def __init__(self,**keyargs):
        pass

class DXData(object):
    def __init__(self,**keyargs):
        pass

