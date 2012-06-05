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

"""
this module handles a dataset object.
A single dataset object or a list of them should be the output of every
future file io routine.

A dataset contains two classes of attributes: required and optional ones
required attributes:
this attributes have to be present and are common to all dataset objects

optional attributes:
this are attributes which are not required but may be constructed by an
io routine.
"""

from matplotlib import pylab as pl

from .. import config

class DataRecord(object):
    def __init__(self):
        self.coldict = {}

    def __str__(self):
        #string representation of the record
        pass



class Dataset(object):
    def __init__(self,setname):
        self.name = setname

        self.coldict = {}
        self.nofrows = 0



    def plot1d(self,*args,**keyargs):
        #check if data is ok for 1D plotting
        #this is the case for the following situations:
        #if only one argument is given
        #1.) the column is a 1D array - the column data is plotted over index space
        #1.) the column is a 2D array - first dimension is row index second
        #                               is data-index. All rows are plotted
        #                               over their index space. A legend with
        #                               the row index is drawn too.
        #
        #if two arguments are given
        #1.) x and y axis are simple 1D arrays
        #2.) x and y axis are 2D arrays
        #

        #===============handling optional arguments======================
        #set a figure object to plot in
        if "fig" in keyargs:
            self.figure = fig
        else:
            self.figure = pl.figure()

        #set a unit for the x-axis
        if "xunit" in keyargs:
            xunit = " "+keyargs["xunit"]
        else:
            xunit = ""

        #set a unit for the y-axis
        if "yunit" in keyargs:
            yunit = " "+keyargs["yunit"]
        else:
            yunit = ""

        #set plot options po=""
        if "po" in keyargs:
            plotopts = keyargs["po"]
        else:
            plotopts = "b-"

        if "po" in keyargs:
            pass

        #build x-axis and y-axis label
        xaxis_name = xname+xunit
        yaxis_name = yname+yunit

        #=============handle the number of input arguments in *args=========
        if len(args)==1:
            xaxis = None
            yaxis = self.coldict[args[0]]
        elif len(args)==2:
            xaxis = self.coldict[args[0]]
            yaxis = self.coldict[args[1]]
        else:
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.io.Dataset: invalid number of arguments")
            return None


        #first situation - x and y are 1D arrays
        if len(xaxis.shape)==1 and len(yaxis.shape)==1:
            pass


        if "ps" in keyargs:
            if keyargs["ps"] == "logx":
                self.plot = pl.semilogx(xaxis,yaxis,plotopts)
            if keyargs["ps"] == "logy":
                self.plot = pl.semilogy(xaxis,yaxis,plotopts)
        else:
            self.plot = figure.plot(xaxis,yaxis,plotopts)


    def plot2d(xnamy,xname,yname):
        pass

    def dump2hdf5(self,h5file):
        pass
