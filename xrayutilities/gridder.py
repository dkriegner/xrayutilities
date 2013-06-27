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
# Copyright (C) 2009-2010,2013 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2009 Mario Keplinger <mario.keplinger@jku.at>
# Copyright (C) 2009-2013 Dominik Kriegner <dominik.kriegner@gmail.com>

import numpy

from . import cxrayutilities
from . import exception
from . import config

unit_dict = {"kb":1024,"mb":1024**2,"gb":1024**3}

def check_array(a,dtype):
    """
    check_array(a,dtype):
    Check if an array fits the requirements for the C-code and returns it 
    back to the callee. Such arrays must be aligned and C_CONTIGUOUS which
    means that they have to follow C-ordering.

    rquired input arguments:
    a ............ array to check
    dtype ........ numpy data type
    """

    return numpy.require(a,dtype=dtype,requirements=["ALIGNED","C_CONTIGUOUS"])

def delta(min_value,max_value,n):
    """
    delta(min_value,max_value,n):
    Compute the stepsize along an axis of a grid. 

    required input arguments:
    min_value ........... axis minimum value
    max_value ........... axis maximum value
    n ................... number of steps
    """

    return (max_value-min_value)/(n-1)

def axis(min_value,max_value,n):
    """
    axis(min_value,max_value,n):
    Compute the a grid axis. 

    required input arguments:
    min_value ........... axis minimum value
    max_value ........... axis maximum value
    n ................... number of steps
    """

    d = delta(min_value,max_value,n)
    a = min_value + d*numpy.arange(0,n)

    return a

def ones(*args):
    """
    matrix(*args):

    Compute ones for matrix generation. The shape is determined by the number of
    input arguments.
    """
    return numpy.ones(args,dtype=numpy.double)


class Gridder(object):
    def __init__(self,**keyargs):
        """
        Basis class for gridders in xrayutilities. A gridder is a function mapping
        irregular spaced data onto a regular grid by binning the data into equally 
        sized elements

        Parameters:
        -----------

         **keyargs (optional):
            nthreads:   number of threads used in the gridding procedure
                        default: 0 -> sequential code is used
        """

        if 'nthreads' in keyargs:
            self.nthreads = keyargs['nthreads']
        else:
            self.nthreads = 0

        self.csize = 0
        self.cunit = 1024**2
        self.flags = 0
        if config.VERBOSITY >= config.INFO_ALL:
            self.flags = self.flags|16 # set verbosity flag


    def SetThreads(self,n):
        """
        SetThreads(n)

        Set the number of threads that should be used for gridding.

        required input arguments:
        n .............. number of threads
        """
        self.nthreads = n

    def SetChunkSize(self,n):
        self.csize = n

    def SetChunkUnit(self,u):
        if u in unit_dict.keys():
            self.cunit = unit_dict[u]
        else:
            raise InputError("Chunk size unit must be one of: kb, mb or gb")

    def Normalize(self,bool):
        self.flags = self.flags^4

    def KeepData(self,bool):
        if not bool==True or bool==False:
            raise TypeError("Keep Data flag must be a boolan value (True/False)!")

        self.keep_data = bool


class Gridder1D(Gridder):
    def __init__(self,nx,**keyargs):
        Gridder.__init__(self,**keyargs)

        self.nx = nx
        self.xmin = 0
        self.xmax = 0

        self.gdata = numpy.zeros(nx,dtype=numpy.double)
        self.gnorm = numpy.zeros(nx,dtype=numpy.double)

    def __get_xaxis(self):
        """
        Returns the xaxis of the gridder
        the returned values correspond to the center of the data bins used by
        the numpy.histogram function
        """
        dx = (self.xmax-self.xmin)/(self.nx)
        ax = self.xmin+dx*numpy.arange(0,self.nx) + dx/2.
        return ax

    def __get_data(self):
        return self.gdata.copy()

    xaxis = property(__get_xaxis)
    data = property(__get_data)

    def Clear(self):
        self.gdata[...] = 0
        self.gnorm[...] = 0

    def __call__(self,*args):
        """
        GridData(x,data):
        Perform gridding on a set of data.

        required input argument:
        x ............... numpy array of arbitrary shape with x positions
        data ............ numpy array of arbitrary shape with data values
        """

        x = args[0]
        data = args[1]
        x = x.reshape(x.size)
        data = data.reshape(data.size)

        if x.size!=data.size:
            raise exception.InputError("XU.Gridder1D: size of given datasets (x,data) is not equal!")

        # require correct aligned memory for input arrays
        x = numpy.require(x,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])
        data = numpy.require(data,dtype=numpy.double,requirements=["ALIGNED","C_CONTIGUOUS"])

        dx = (self.xmax-self.xmin)/(self.nx) # no -1 here to be consistent with numpy.histogram

        # use only non-NaN data values
        mask = numpy.invert(numpy.isnan(data))
        ldata = data[mask]
        lx = x[mask]

        self.xmin = lx.min()
        self.xmax = lx.max()

        # grid the data using numpy histogram
        self.gdata,bins = numpy.histogram(lx,weights=ldata,bins=self.nx)
        self.gnorm,bins = numpy.histogram(lx,bins=self.nx)

        if self.flags != self.flags|4:
            self.gdata[self.gnorm!=0] /= self.gnorm[self.gnorm!=0].astype(numpy.float)




